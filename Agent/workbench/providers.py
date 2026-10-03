import json
import math

import httpx

from .config import Settings


class ProviderError(RuntimeError):
    pass


class VLLM:
    """OpenAI-compatible HTTP protocol; inference runs in a separate process."""

    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self.transport = transport

    def client(self, base, key):
        return httpx.AsyncClient(
            base_url=base.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=httpx.Timeout(self.settings.request_timeout, connect=10),
            transport=self.transport,
        )

    async def models(self, embeddings=False):
        s = self.settings
        base, key = (
            (s.embedding_base_url, s.embedding_api_key)
            if embeddings
            else (s.llm_base_url, s.llm_api_key)
        )
        async with self.client(base, key) as client:
            response = await client.get("models")
            self.check(response)
            return [m["id"] for m in response.json().get("data", [])]

    @staticmethod
    def check(response):
        if response.is_error:
            # Do not expose upstream credentials, stack traces, or request payloads.
            raise ProviderError(
                f"推理服务返回 HTTP {response.status_code}；请检查模型名称、API key、上下文长度和 vLLM 日志"
            )

    async def stream(self, messages, tools, force_final=False):
        s = self.settings
        body = {
            "model": s.llm_model,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": s.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            body.update(
                tools=tools,
                tool_choice="none" if force_final else "auto",
                parallel_tool_calls=False,
            )
        calls, content, finish, saw_done = {}, "", None, False
        try:
            async with self.client(s.llm_base_url, s.llm_api_key) as client:
                async with client.stream("POST", "chat/completions", json=body) as response:
                    self.check(response)
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            saw_done = True
                            break
                        if not raw:
                            continue
                        packet = json.loads(raw)
                        if packet.get("error"):
                            raise ProviderError("vLLM 流式响应报告错误，请检查推理日志")
                        if packet.get("usage"):
                            yield {"kind": "usage", "data": packet["usage"]}
                        for choice in packet.get("choices", []):
                            if choice.get("index", 0) != 0:
                                continue
                            finish = choice.get("finish_reason") or finish
                            delta = choice.get("delta", {})
                            # reasoning_content is deliberately not displayed or stored.
                            text = delta.get("content") or ""
                            if text:
                                content += text
                                if len(content) > 100000:
                                    raise ProviderError("模型输出超过限制")
                                yield {"kind": "delta", "data": {"text": text}}
                            for tc in delta.get("tool_calls") or []:
                                index = tc.get("index", 0)
                                if not isinstance(index, int) or not 0 <= index < 4:
                                    raise ProviderError("模型返回了过多工具调用")
                                item = calls.setdefault(
                                    index,
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    },
                                )
                                if tc.get("id"):
                                    item["id"] = tc["id"]
                                function = tc.get("function", {})
                                for key in ("name", "arguments"):
                                    item["function"][key] += function.get(key) or ""
                                if len(item["function"]["arguments"]) > 16000:
                                    raise ProviderError("工具参数超过限制")
            if not saw_done or not finish:
                raise ProviderError("推理连接提前结束，请重试")
            if finish == "length":
                raise ProviderError("模型输出达到 token 上限；请缩短问题或提高 AW_MAX_TOKENS")
            if calls and force_final:
                raise ProviderError("模型在最终回答阶段仍请求工具")
            if not calls and not content.strip():
                raise ProviderError("模型没有返回可显示的回答")
            ordered = [calls[i] for i in sorted(calls)]
            for i, call in enumerate(ordered):
                call["id"] = call["id"] or f"call_{i}"
            yield {
                "kind": "completion",
                "data": {
                    "role": "assistant",
                    "content": content or None,
                    **({"tool_calls": ordered} if ordered else {}),
                },
            }
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderError("无法完成 vLLM 请求，请检查服务地址、网络、超时和服务日志") from exc

    async def embed(self, texts, query=False):
        s = self.settings
        if query:
            texts = [
                "Instruct: Given a search query, retrieve relevant passages that answer the query"
                f"\nQuery: {text}"
                for text in texts
            ]
        vectors = []
        try:
            async with self.client(s.embedding_base_url, s.embedding_api_key) as client:
                for offset in range(0, len(texts), 32):
                    batch = texts[offset : offset + 32]
                    response = await client.post(
                        "embeddings",
                        json={
                            "model": s.embedding_model,
                            "input": batch,
                            "encoding_format": "float",
                        },
                    )
                    self.check(response)
                    data = sorted(response.json()["data"], key=lambda x: x["index"])
                    if [row["index"] for row in data] != list(range(len(batch))):
                        raise ProviderError("Embedding 返回数量或索引不正确")
                    for row in data:
                        vector = [float(v) for v in row["embedding"]]
                        norm = math.sqrt(sum(v * v for v in vector))
                        if not vector or not math.isfinite(norm) or norm == 0:
                            raise ProviderError("Embedding 返回无效向量")
                        vectors.append([v / norm for v in vector])
            if len({len(v) for v in vectors}) > 1:
                raise ProviderError("Embedding 向量维度不一致")
            return vectors
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ProviderError("无法获得有效 Embedding，请检查向量服务") from exc
