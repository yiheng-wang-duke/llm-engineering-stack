import asyncio
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

from .db import now, uid


class IndexMismatch(ValueError):
    pass


def tokenize(text):
    words = re.findall(r"[a-z0-9_]+", text.lower())
    for segment in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(segment)
        words.extend(segment[i : i + 2] for i in range(len(segment) - 1))
    return words


def bm25(query, texts):
    query_tokens = set(tokenize(query))
    tokens = [Counter(tokenize(text)) for text in texts]
    lengths = [sum(t.values()) for t in tokens]
    avg = sum(lengths) / max(len(lengths), 1) or 1
    df = Counter(word for item in tokens for word in item if word in query_tokens)
    result = []
    for index, counter in enumerate(tokens):
        score = 0.0
        for token in query_tokens:
            tf = counter.get(token, 0)
            if tf:
                idf = math.log(1 + (len(tokens) - df[token] + 0.5) / (df[token] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[index] / avg))
        result.append(score)
    return result


def split_text(text, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Invalid chunk size/overlap")
    text = text.replace("\x00", "").strip()
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + size // 2, end),
                text.rfind("。", start + size // 2, end),
                text.rfind(". ", start + size // 2, end),
            )
            if boundary > start:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            yield part
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def parse_document(name, raw, max_chars):
    suffix = Path(name).suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".json"}:
        try:
            pages = [(None, raw.decode("utf-8-sig"))]
        except UnicodeDecodeError as exc:
            raise ValueError("文本文件请使用 UTF-8 编码") from exc
    elif suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError("请先解密 PDF")
            if len(reader.pages) > 200:
                raise ValueError("PDF 最多支持 200 页")
            pages, total = [], 0
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                total += len(text)
                if total > max_chars:
                    raise ValueError("文档提取文本超过字符限制")
                pages.append((number, text))
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("PDF 解析失败，请确认文件完整") from exc
    else:
        raise ValueError("支持 .txt / .md / .pdf / .csv / .json")
    if sum(len(text) for _, text in pages) > max_chars:
        raise ValueError("文档提取文本超过字符限制")
    if not any(text.strip() for _, text in pages):
        raise ValueError("文档没有可提取文本；扫描 PDF 请先执行 OCR")
    return pages


class Knowledge:
    def __init__(self, store, settings, provider):
        self.store, self.settings, self.provider = store, settings, provider
        self.write_lock = asyncio.Lock()

    @property
    def embedding_key(self):
        s = self.settings
        return (
            "bm25"
            if s.embedding_backend == "bm25"
            else (f"vllm:{s.embedding_base_url.rstrip('/')}:{s.embedding_model}")
        )

    async def ingest(self, name, raw):
        s = self.settings
        # Original names are labels only, never filesystem paths.
        name = name.replace("\\", "/").rsplit("/", 1)[-1][:160] or "document.txt"
        digest = hashlib.sha256(raw).hexdigest()
        async with self.write_lock:
            existing = self.store.rows("SELECT id FROM documents WHERE digest=?", (digest,))
            if existing:
                return {"id": existing[0]["id"], "duplicate": True}
            pages = await asyncio.to_thread(parse_document, name, raw, s.max_document_chars)
            chunks = [
                {"id": uid(), "content": text, "page": page}
                for page, content in pages
                for text in split_text(content, s.chunk_size, s.chunk_overlap)
            ]
            count = self.store.rows("SELECT COUNT(*) AS n FROM chunks")[0]["n"]
            if count + len(chunks) > s.max_chunks:
                raise ValueError("知识库分块数达到上限，请删除部分文档")
            vectors = (
                await self.provider.embed([c["content"] for c in chunks])
                if (s.embedding_backend == "vllm")
                else [None] * len(chunks)
            )
            doc_id = uid()
            with self.store.connect() as conn:
                conn.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                    (
                        doc_id,
                        name,
                        digest,
                        json.dumps(pages, ensure_ascii=False),
                        self.embedding_key,
                        now(),
                    ),
                )
                conn.executemany(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                    [
                        (
                            c["id"],
                            doc_id,
                            i,
                            c["content"],
                            c["page"],
                            json.dumps(v) if v is not None else None,
                        )
                        for i, (c, v) in enumerate(zip(chunks, vectors))
                    ],
                )
            return {"id": doc_id, "name": name, "chunks": len(chunks), "duplicate": False}

    async def reindex(self):
        async with self.write_lock:
            chunks = self.store.rows("SELECT id,content FROM chunks ORDER BY id")
            vectors = (
                await self.provider.embed([c["content"] for c in chunks])
                if (self.settings.embedding_backend == "vllm")
                else [None] * len(chunks)
            )
            # Commit the complete new index atomically; failures keep the old index intact.
            with self.store.connect() as conn:
                conn.executemany(
                    "UPDATE chunks SET vector=? WHERE id=?",
                    [
                        (json.dumps(v) if v is not None else None, c["id"])
                        for c, v in zip(chunks, vectors)
                    ],
                )
                conn.execute("UPDATE documents SET embedding_key=?", (self.embedding_key,))
            return {"chunks": len(chunks), "embedding_key": self.embedding_key}

    async def delete(self, doc_id):
        async with self.write_lock:
            return self.store.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    async def search(self, query, top_k=None):
        chunks = self.store.rows("""SELECT c.*,d.name,d.embedding_key FROM chunks c
            JOIN documents d ON d.id=c.document_id ORDER BY c.id""")
        if not chunks:
            return []
        if any(c["embedding_key"] != self.embedding_key for c in chunks):
            raise IndexMismatch("索引配置已变更，请在知识库中点击「重建索引」")
        lexical = await asyncio.to_thread(bm25, query, [c["content"] for c in chunks])
        lexical_order = sorted(
            [i for i, score in enumerate(lexical) if score > 0],
            key=lambda i: lexical[i],
            reverse=True,
        )[:30]
        scores = {i: 1 / (60 + rank) for rank, i in enumerate(lexical_order, 1)}
        if self.settings.embedding_backend == "vllm":
            vector = (await self.provider.embed([query], query=True))[0]
            dense = []
            for i, chunk in enumerate(chunks):
                stored = json.loads(chunk["vector"] or "[]")
                if len(stored) != len(vector):
                    raise IndexMismatch("向量维度已变更，请重建索引")
                similarity = sum(a * b for a, b in zip(vector, stored))
                if similarity > 0:
                    dense.append((i, similarity))
            for rank, (i, _) in enumerate(sorted(dense, key=lambda x: -x[1])[:30], 1):
                scores[i] = scores.get(i, 0) + 1 / (60 + rank)
        ranked = sorted(scores, key=lambda i: -scores[i])[: top_k or self.settings.top_k]
        return [
            {
                "id": chunks[i]["id"],
                "document_id": chunks[i]["document_id"],
                "name": chunks[i]["name"],
                "ordinal": chunks[i]["ordinal"],
                "page": chunks[i]["page"],
                "content": chunks[i]["content"],
                "score": round(scores[i], 6),
            }
            for i in ranked
        ]
