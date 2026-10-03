"""Real HTTP smoke test against a running app; creates temporary test records and cleans up."""

import argparse
import json
import os

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--require-vllm", action="store_true")
    args = parser.parse_args()
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.getenv("AW_API_TOKEN", "")
    headers = {"Authorization": "Bearer " + token} if token else {}
    session_id = doc_id = memory_id = None
    with httpx.Client(base_url=args.url, headers=headers, timeout=180) as client:

        def request(method, path, **kwargs):
            r = client.request(method, "/api" + path, **kwargs)
            r.raise_for_status()
            return r

        try:
            config = request("GET", "/config").json()
            if args.require_vllm and config["mode"] != "vllm":
                raise RuntimeError("Expected real vLLM mode")
            if args.require_vllm:
                health = request("GET", "/health").json()
                assert health["inference"]["ready"], health
            session_id = request("POST", "/sessions", json={"title": "Smoke test"}).json()["id"]
            import uuid

            marker = "smoke-" + uuid.uuid4().hex[:12]
            memory_id = request(
                "POST", "/memories", json={"content": marker + "：我喜欢中文回答"}
            ).json()["id"]
            doc_id = request(
                "POST",
                "/documents",
                files={
                    "file": ("smoke.md", f"{marker} 的项目代号是松林，部署端口为 8080。".encode())
                },
            ).json()["id"]
            hits = request("POST", "/knowledge/search", json={"query": marker}).json()
            assert hits and hits[0]["document_id"] == doc_id
            response = request(
                "POST",
                f"/sessions/{session_id}/chat",
                json={"message": marker + " 的项目代号是什么？引用知识库来源。"},
            )
            events = []
            for frame in response.text.split("\n\n"):
                kind, data = None, None
                for line in frame.splitlines():
                    if line.startswith("event: "):
                        kind = line[7:]
                    elif line.startswith("data: "):
                        data = json.loads(line[6:])
                if kind:
                    events.append((kind, data))
            assert not any(k == "error" for k, _ in events), events
            assert any(k == "done" and d["sources"] for k, d in events), events
            messages = request("GET", f"/sessions/{session_id}").json()["messages"]
            assert len(messages) == 2
            print("PASS: HTTP upload → retrieval → memory → chat/SSE → persisted history")
            print("Mode:", config["mode"])
        finally:
            for path in [
                f"/sessions/{session_id}" if session_id else None,
                f"/documents/{doc_id}" if doc_id else None,
                f"/memories/{memory_id}" if memory_id else None,
            ]:
                if path:
                    request("DELETE", path)


if __name__ == "__main__":
    main()
