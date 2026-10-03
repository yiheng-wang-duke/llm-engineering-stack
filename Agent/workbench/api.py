import asyncio
import contextlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent import Agent
from .config import Settings
from .db import Store, now
from .providers import VLLM, ProviderError
from .rag import IndexMismatch, Knowledge

log = logging.getLogger(__name__)


class TextInput(BaseModel):
    content: str = Field(min_length=1, max_length=1000)

    @field_validator("content")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("内容不能为空")
        return value.strip()


class ChatInput(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    use_rag: bool = True
    use_memory: bool = True

    @field_validator("message")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


class SessionInput(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=80)


class SearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)


class BodyLimit:
    """Bound the body even when a client uses chunked transfer encoding."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.limit:
                    raise HTTPException(413, "请求超过上传大小限制")
            return message

        await self.app(scope, bounded_receive, send)


def create_app(settings=None, provider=None):
    settings = settings or Settings()
    store = Store(settings.data_dir / "workbench.sqlite3")
    provider = provider or VLLM(settings)
    knowledge = Knowledge(store, settings, provider)
    agent = Agent(store, settings, knowledge, provider)
    busy = set()

    @asynccontextmanager
    async def lifespan(app):
        store.execute(
            "UPDATE runs SET status='interrupted',finished_at=? WHERE status='running'", (now(),)
        )
        yield

    app = FastAPI(title="Qwen Agent Workbench", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.agent = store, agent
    app.state.knowledge, app.state.busy = knowledge, busy
    app.add_middleware(BodyLimit, limit=settings.max_upload_mb * 1024 * 1024 + 65536)

    @app.middleware("http")
    async def access(request, call_next):
        if request.url.path.startswith("/api/"):
            expected = settings.api_token
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
            if expected and not hmac.compare_digest(supplied.encode(), expected.encode()):
                return JSONResponse({"detail": "需要有效的访问令牌"}, status_code=401)
            origin = request.headers.get("origin")
            if (
                request.method not in {"GET", "HEAD", "OPTIONS"}
                and origin
                and (urlparse(origin).netloc != request.headers.get("host"))
            ):
                return JSONResponse({"detail": "不允许跨来源写入"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
            )
        return response

    @app.exception_handler(IndexMismatch)
    async def index_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ProviderError)
    async def provider_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    def require_session(session_id):
        if not store.rows("SELECT id FROM sessions WHERE id=?", (session_id,)):
            raise HTTPException(404, "会话不存在")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/api/config")
    async def config():
        return {
            "mode": settings.mode,
            "model": settings.llm_model,
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model
            if settings.embedding_backend == "vllm"
            else None,
            "max_steps": settings.max_steps,
            "max_upload_mb": settings.max_upload_mb,
            "tools": ["search_knowledge", "recall_memory", "calculate", "current_time"],
        }

    @app.get("/api/health")
    async def health():
        async def check(embedding=False):
            try:
                models = await asyncio.wait_for(provider.models(embedding), timeout=6)
                configured = settings.embedding_model if embedding else settings.llm_model
                return {
                    "ready": configured in models,
                    "models": models,
                    "detail": "" if configured in models else "配置的模型名称未出现在 /v1/models",
                }
            except Exception:
                return {"ready": False, "detail": "服务未就绪或模型不可访问"}

        inference = (
            {"ready": False, "detail": "离线演示：没有调用模型"}
            if (settings.mode == "demo")
            else await check()
        )
        embedding = (
            await check(True)
            if settings.embedding_backend == "vllm"
            else {"ready": True, "detail": "BM25 关键词检索"}
        )
        return {"inference": inference, "embedding": embedding}

    @app.get("/api/sessions")
    async def sessions():
        return store.sessions()

    @app.post("/api/sessions", status_code=201)
    async def new_session(body: SessionInput):
        return store.create_session(body.title)

    @app.get("/api/sessions/{session_id}")
    async def session(session_id: str):
        require_session(session_id)
        return {
            "messages": store.messages(session_id),
            "runs": store.rows(
                "SELECT * FROM runs WHERE session_id=? ORDER BY created_at DESC", (session_id,)
            ),
        }

    @app.patch("/api/sessions/{session_id}")
    async def rename_session(session_id: str, body: SessionInput):
        require_session(session_id)
        store.execute(
            "UPDATE sessions SET title=?,updated_at=? WHERE id=?", (body.title, now(), session_id)
        )
        return {"ok": True}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        require_session(session_id)
        if session_id in busy:
            raise HTTPException(409, "请先停止当前生成")
        store.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return {"ok": True}

    @app.get("/api/sessions/{session_id}/export")
    async def export(session_id: str):
        require_session(session_id)
        return JSONResponse(
            {"session_id": session_id, "messages": store.messages(session_id)},
            headers={"Content-Disposition": f'attachment; filename="chat-{session_id}.json"'},
        )

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        runs = store.rows("SELECT * FROM runs WHERE id=?", (run_id,))
        if not runs:
            raise HTTPException(404, "执行记录不存在")
        events = store.rows(
            "SELECT kind,payload,created_at FROM events WHERE run_id=? ORDER BY id", (run_id,)
        )
        for event in events:
            event["data"] = json.loads(event.pop("payload"))
        return {**runs[0], "events": events}

    @app.post("/api/sessions/{session_id}/chat")
    async def chat(session_id: str, body: ChatInput, request: Request):
        require_session(session_id)
        if session_id in busy:
            raise HTTPException(409, "此会话已有正在运行的请求")
        busy.add(session_id)
        try:
            run_id = store.start_run(session_id, body.message)
        except Exception:
            busy.discard(session_id)
            raise

        async def stream():
            queue = asyncio.Queue(maxsize=128)

            async def produce():
                try:
                    async for event in agent.run(
                        run_id, session_id, body.message, body.use_rag, body.use_memory
                    ):
                        if event["kind"] != "delta":
                            store.event(run_id, event["kind"], event["data"])
                        await queue.put(event)
                except asyncio.CancelledError:
                    store.fail_run(run_id, "用户停止或连接断开", "cancelled")
                    raise
                except Exception as exc:
                    log.exception("Agent run %s failed", run_id)
                    detail = (
                        str(exc)
                        if isinstance(exc, (ValueError, ProviderError))
                        else ("执行失败，请查看服务日志和运行编号")
                    )
                    store.fail_run(run_id, detail)
                    event = {"kind": "error", "data": {"message": detail, "run_id": run_id}}
                    store.event(run_id, "error", event["data"])
                    await queue.put(event)
                finally:
                    # Don't block cancellation on a full queue.
                    if not asyncio.current_task().cancelling():
                        await queue.put(None)

            task = asyncio.create_task(produce())
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=10)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield ": heartbeat\n\n"
                        continue
                    if event is None:
                        break
                    yield f"event: {event['kind']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                busy.discard(session_id)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/memories")
    async def memories():
        return store.memories()

    @app.post("/api/memories", status_code=201)
    async def add_memory(body: TextInput):
        return store.add_memory(body.content)

    @app.delete("/api/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        if not store.execute("DELETE FROM memories WHERE id=?", (memory_id,)):
            raise HTTPException(404, "记忆不存在")
        return {"ok": True}

    @app.get("/api/documents")
    async def documents():
        return store.documents()

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile = File(...)):
        try:
            raw = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            if len(raw) > settings.max_upload_mb * 1024 * 1024:
                raise HTTPException(413, "文件超过上传大小限制")
            return await knowledge.ingest(file.filename or "document.txt", raw)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()

    @app.get("/api/documents/{document_id}")
    async def document(document_id: str):
        docs = store.rows("SELECT id,name,created_at FROM documents WHERE id=?", (document_id,))
        if not docs:
            raise HTTPException(404, "文档不存在")
        chunks = store.rows(
            "SELECT id,ordinal,content,page FROM chunks WHERE document_id=? ORDER BY ordinal",
            (document_id,),
        )
        return {**docs[0], "chunks": chunks}

    @app.delete("/api/documents/{document_id}")
    async def delete_document(document_id: str):
        if not await knowledge.delete(document_id):
            raise HTTPException(404, "文档不存在")
        return {"ok": True}

    @app.post("/api/knowledge/search")
    async def search(body: SearchInput):
        return await knowledge.search(body.query)

    @app.post("/api/knowledge/reindex")
    async def reindex():
        return await knowledge.reindex()

    static = Path(__file__).parent / "static"
    static.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    async def index():
        return FileResponse(static / "index.html")

    return app
