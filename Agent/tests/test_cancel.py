import asyncio
import socket
import threading
import time

import httpx
import uvicorn

from workbench.api import create_app


class SlowProvider:
    async def stream(self, messages, tools, force_final=False):
        await asyncio.sleep(60)
        yield {"kind": "delta", "data": {"text": "too late"}}


def test_real_disconnect_cancels_run_and_releases_session(settings):
    settings.mode = "vllm"
    app = create_app(settings, SlowProvider())
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started

        async def disconnect():
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                session = (await client.post("/api/sessions", json={})).json()["id"]
                async with client.stream(
                    "POST", f"/api/sessions/{session}/chat", json={"message": "slow request"}
                ) as response:
                    async for line in response.aiter_lines():
                        if line == "event: response_start":
                            break
                for _ in range(100):
                    if session not in app.state.busy:
                        break
                    await asyncio.sleep(0.02)
                assert session not in app.state.busy
                detail = (await client.get(f"/api/sessions/{session}")).json()
                assert detail["runs"][0]["status"] == "cancelled"
                assert detail["messages"] == []

        asyncio.run(disconnect())
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
