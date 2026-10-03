import asyncio
import io
import json

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from workbench.api import create_app
from workbench.db import Store
from workbench.memory import build_history, fit_context
from workbench.rag import IndexMismatch, Knowledge, bm25, parse_document, split_text
from workbench.tools import calculate


def events(response):
    assert response.status_code == 200, response.text
    result = []
    for frame in response.text.split("\n\n"):
        lines = frame.splitlines()
        kind = next((s[7:] for s in lines if s.startswith("event: ")), None)
        payload = next((s[6:] for s in lines if s.startswith("data: ")), None)
        if kind:
            result.append((kind, json.loads(payload)))
    return result


def new_session(client):
    return client.post("/api/sessions", json={}).json()["id"]


def test_complete_demo_workflow_and_persistence(client, settings):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/api/config").json()["mode"] == "demo"
    document = client.post(
        "/api/documents",
        files={"file": ("../../手册.md", "松林项目使用 FastAPI，部署端口是 8080。".encode())},
    ).json()
    assert document["name"] == "手册.md"
    duplicate = client.post(
        "/api/documents",
        files={"file": ("same.md", "松林项目使用 FastAPI，部署端口是 8080。".encode())},
    ).json()
    assert duplicate["duplicate"] is True
    hits = client.post("/api/knowledge/search", json={"query": "松林项目部署端口"}).json()
    assert hits[0]["document_id"] == document["id"]
    first = new_session(client)
    recorded = events(
        client.post(
            f"/api/sessions/{first}/chat", json={"message": "/remember 我喜欢中文与 Python 示例"}
        )
    )
    assert "memory_saved" in [kind for kind, _ in recorded]
    second = new_session(client)
    response = events(
        client.post(f"/api/sessions/{second}/chat", json={"message": "松林项目部署端口是什么？"})
    )
    done = next(data for kind, data in response if kind == "done")
    assert "离线演示" in done["answer"] and "8080" in done["answer"]
    assert done["sources"][0]["label"] == "S1"
    assert any(kind == "memory" and data["items"] for kind, data in response)
    messages = client.get(f"/api/sessions/{second}").json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    run = client.get("/api/runs/" + messages[-1]["run_id"]).json()
    assert run["status"] == "completed" and run["events"]
    exported = client.get(f"/api/sessions/{second}/export").json()
    assert exported["messages"][-1]["sources"]
    with TestClient(create_app(settings)) as restarted:
        assert len(restarted.get("/api/memories").json()) == 1
        assert len(restarted.get(f"/api/sessions/{second}").json()["messages"]) == 2
    assert client.delete("/api/documents/" + document["id"]).status_code == 200
    assert client.post("/api/knowledge/search", json={"query": "松林"}).json() == []
    assert client.delete(f"/api/sessions/{second}").status_code == 200
    assert client.get("/api/runs/" + messages[-1]["run_id"]).status_code == 404


def test_api_validation_and_auth(settings):
    settings.api_token = "test-secret"
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/config").status_code == 401
        assert client.get("/healthz").status_code == 200
        client.headers["Authorization"] = "Bearer test-secret"
        assert "test-secret" not in client.get("/api/config").text
        assert client.post("/api/memories", json={"content": "  "}).status_code == 422
        assert (
            client.post(
                "/api/sessions", json={}, headers={"Origin": "https://external.example"}
            ).status_code
            == 403
        )
        assert client.post("/api/documents", files={"file": ("bad.exe", b"bad")}).status_code == 400
        assert (
            client.post("/api/documents", files={"file": ("bad.txt", b"\xff")}).status_code == 400
        )
        assert client.get("/api/sessions/missing").status_code == 404


def test_upload_limits(settings):
    settings.max_upload_mb = 1
    with TestClient(create_app(settings)) as client:
        assert (
            client.post(
                "/api/documents", files={"file": ("large.txt", b"x" * (1024 * 1024 + 1))}
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/api/documents", files={"file": ("huge.txt", b"x" * (2 * 1024 * 1024))}
            ).status_code
            == 413
        )
        assert client.get("/api/documents").json() == []


def test_session_busy_and_cancel_state(client, app):
    session = new_session(client)
    app.state.busy.add(session)
    assert client.post(f"/api/sessions/{session}/chat", json={"message": "hi"}).status_code == 409
    assert client.delete(f"/api/sessions/{session}").status_code == 409
    app.state.busy.clear()
    run_id = app.state.store.start_run(session, "interrupted request")
    app.state.store.fail_run(run_id, "cancelled", "cancelled")
    assert (
        app.state.store.rows("SELECT status FROM runs WHERE id=?", (run_id,))[0]["status"]
        == "cancelled"
    )


def test_memory_idempotence_delete_and_disabled(client):
    one = client.post("/api/memories", json={"content": "prefer Python"}).json()
    two = client.post("/api/memories", json={"content": "prefer Python"}).json()
    assert one["id"] == two["id"]
    session = new_session(client)
    response = events(
        client.post(
            f"/api/sessions/{session}/chat",
            json={"message": "hello", "use_memory": False, "use_rag": False},
        )
    )
    assert next(data for kind, data in response if kind == "memory")["items"] == []
    assert client.delete("/api/memories/" + one["id"]).status_code == 200
    assert client.get("/api/memories").json() == []


@pytest.mark.parametrize(
    "expression,result",
    [
        ("(128 * 24 + 960) / 12", 336),
        ("-2 ** 3", -8),
        ("10 % 3", 1),
        ("2**-2", 0.25),
    ],
)
def test_calculator(expression, result):
    assert calculate(expression) == result


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd')",
        "2**999",
        "1/0",
        "1e999",
        "[1][0]",
        "(1).__class__",
        "9**9**9",
    ],
)
def test_calculator_rejects_unsafe_or_unbounded(expression):
    with pytest.raises((ValueError, ArithmeticError, SyntaxError)):
        calculate(expression)


def test_chunking_and_bilingual_bm25():
    text = "这是中文知识。\nPython FastAPI agent memory.\n" * 100
    chunks = list(split_text(text, 140, 30))
    assert len(chunks) > 1 and all(0 < len(c) <= 140 for c in chunks)
    assert bm25("FastAPI", ["FastAPI Python", "tomatoes"])[0] > 0
    assert bm25("长期记忆", ["用户长期记忆", "推理服务"])[0] > 0
    assert bm25("unrelated", ["FastAPI Python"]) == [0]
    with pytest.raises(ValueError):
        list(split_text(text, 10, 10))


def test_pdf_empty_and_corrupt():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    output = io.BytesIO()
    writer.write(output)
    with pytest.raises(ValueError, match="OCR"):
        parse_document("scan.pdf", output.getvalue(), 10000)
    with pytest.raises(ValueError, match="PDF"):
        parse_document("bad.pdf", b"invalid", 10000)


def test_history_budget_preserves_pairs():
    messages = [
        {"role": r, "content": str(i) * 200} for i in range(20) for r in ("user", "assistant")
    ]
    recent, recap = build_history(messages, 3, 1000)
    assert len(recent) % 2 == 0 and recent[0]["role"] == "user"
    assert sum(len(m["content"]) for m in recent) <= 1000
    assert recap and len(recap) <= 1000
    context = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old" * 100},
        {"role": "assistant", "content": "old" * 100},
        {"role": "user", "content": "latest"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": "observation" * 100},
    ]
    trimmed = fit_context(context, 300)
    assert [m["role"] for m in trimmed] == ["system", "user"]
    assert trimmed[-1]["content"] == "latest"


class Embeddings:
    def __init__(self):
        self.fail = False
        self.dimension = 2

    async def embed(self, texts, query=False):
        if self.fail:
            raise RuntimeError("embedding unavailable")
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in texts]


def test_hybrid_reindex_atomicity_and_dimensions(settings):
    async def scenario():
        store = Store(settings.data_dir / "test.sqlite")
        provider = Embeddings()
        knowledge = Knowledge(store, settings, provider)
        await knowledge.ingest("first.md", b"semantic retrieval document")
        settings.embedding_backend = "vllm"
        with pytest.raises(IndexMismatch):
            await knowledge.search("paraphrase")
        provider.fail = True
        with pytest.raises(RuntimeError):
            await knowledge.reindex()
        assert store.documents()[0]["embedding_key"] == "bm25"
        provider.fail = False
        await knowledge.reindex()
        assert (await knowledge.search("paraphrase"))[0]["name"] == "first.md"
        provider.dimension = 3
        with pytest.raises(IndexMismatch):
            await knowledge.search("query")
        provider.fail = True
        with pytest.raises(RuntimeError):
            await knowledge.ingest("second.md", b"new document")
        assert len(store.documents()) == 1

    asyncio.run(scenario())
