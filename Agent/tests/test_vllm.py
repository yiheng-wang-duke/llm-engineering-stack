import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from workbench.api import create_app
from workbench.providers import VLLM, ProviderError

from .test_stack import events, new_session


def sse(*packets):
    return (
        "".join("data: " + json.dumps(p, ensure_ascii=False) + "\n\n" for p in packets)
        + "data: [DONE]\n\n"
    )


def packet(delta=None, finish=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def test_real_http_protocol_tool_loop(settings):
    settings.mode = "vllm"
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == settings.llm_model
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        assert body["stream"] is True
        if len(requests) == 1:
            content = sse(
                packet(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "calculate", "arguments": '{"expression":'},
                            }
                        ]
                    }
                ),
                packet({"tool_calls": [{"index": 0, "function": {"arguments": '"2+3"}'}}]}),
                packet(finish="tool_calls"),
            )
        else:
            assert body["messages"][-1]["role"] == "tool"
            assert json.loads(body["messages"][-1]["content"])["result"]["result"] == 5
            content = sse(
                packet({"content": "结果是"}),
                packet({"content": " 5。[S1]"}),
                packet(finish="stop"),
                {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 8}},
            )
        return httpx.Response(200, text=content, headers={"Content-Type": "text/event-stream"})

    provider = VLLM(settings, transport=httpx.MockTransport(handler))
    with TestClient(create_app(settings, provider)) as client:
        client.post("/api/documents", files={"file": ("math.md", "计算规则：2+3=5".encode())})
        session = new_session(client)
        result = events(client.post(f"/api/sessions/{session}/chat", json={"message": "计算 2+3"}))
        assert next(d for k, d in result if k == "done")["answer"] == "结果是 5。[S1]"
        assert [k for k, _ in result].count("tool_call") == 1
        assert any(k == "usage" for k, _ in result)
        assert len(requests) == 2
        assert len(client.get(f"/api/sessions/{session}").json()["messages"]) == 2


@pytest.mark.parametrize(
    "body",
    [
        "data: not-json\n\n",
        sse(packet({"content": "unfinished"}), packet(finish="length")),
        "data: " + json.dumps(packet({"content": "broken connection"})) + "\n\n",
    ],
)
def test_protocol_errors_do_not_save_fake_success(settings, body):
    settings.mode = "vllm"
    provider = VLLM(settings, httpx.MockTransport(lambda _: httpx.Response(200, text=body)))
    with TestClient(create_app(settings, provider)) as client:
        session = new_session(client)
        result = events(client.post(f"/api/sessions/{session}/chat", json={"message": "hello"}))
        assert any(k == "error" for k, _ in result)
        assert not any(k == "done" for k, _ in result)
        data = client.get(f"/api/sessions/{session}").json()
        assert data["messages"] == []
        assert data["runs"][0]["status"] == "failed"
        assert session not in client.app.state.busy


def test_tool_errors_and_budget(settings):
    settings.mode = "vllm"
    settings.max_steps = 3
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body["tool_choice"] == "none":
            assert "error" in body["messages"][-1]["content"]
            return httpx.Response(
                200, text=sse(packet({"content": "工具参数无效。"}), packet(finish="stop"))
            )
        return httpx.Response(
            200,
            text=sse(
                packet(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "bad",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"expression":"2**1000"}',
                                },
                            }
                        ]
                    }
                ),
                packet(finish="tool_calls"),
            ),
        )

    with TestClient(create_app(settings, VLLM(settings, httpx.MockTransport(handler)))) as client:
        session = new_session(client)
        result = events(client.post(f"/api/sessions/{session}/chat", json={"message": "calculate"}))
        assert len(bodies) == 3 and bodies[-1]["tool_choice"] == "none"
        assert any(k == "tool_result" and not d["ok"] for k, d in result)
        assert any(k == "done" for k, _ in result)


def test_disabled_tools_are_not_executable(settings):
    settings.mode = "vllm"
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert "recall_memory" not in [t["function"]["name"] for t in body["tools"]]
        if calls == 1:
            return httpx.Response(
                200,
                text=sse(
                    packet(
                        {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "a",
                                    "function": {
                                        "name": "recall_memory",
                                        "arguments": '{"query":"secret"}',
                                    },
                                }
                            ]
                        }
                    ),
                    packet(finish="tool_calls"),
                ),
            )
        assert "已被用户禁用" in body["messages"][-1]["content"]
        return httpx.Response(
            200, text=sse(packet({"content": "无法读取记忆"}), packet(finish="stop"))
        )

    with TestClient(create_app(settings, VLLM(settings, httpx.MockTransport(handler)))) as client:
        client.post("/api/memories", json={"content": "secret-memory"})
        session = new_session(client)
        result = events(
            client.post(
                f"/api/sessions/{session}/chat",
                json={"message": "hi", "use_memory": False, "use_rag": False},
            )
        )
        assert any(k == "done" for k, _ in result)
        assert not any("secret-memory" in json.dumps(d) for _, d in result)


def test_embedding_protocol_query_instruction_and_order(settings):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [3.0, 4.0]}
                    for i in reversed(range(len(body["input"])))
                ]
            },
        )

    provider = VLLM(settings, httpx.MockTransport(handler))
    assert asyncio.run(provider.embed(["a", "b"])) == [[0.6, 0.8], [0.6, 0.8]]
    asyncio.run(provider.embed(["question"], query=True))
    assert bodies[1]["input"][0].startswith("Instruct:")
    assert bodies[0]["input"] == ["a", "b"]


def test_embedding_rejects_missing_vectors(settings):
    provider = VLLM(settings, httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []})))
    with pytest.raises(ProviderError):
        asyncio.run(provider.embed(["a"]))
