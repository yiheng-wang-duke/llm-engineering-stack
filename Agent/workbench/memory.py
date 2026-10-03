from .rag import bm25


def recall(store, query, limit=5):
    items = store.memories()
    scores = bm25(query, [item["content"] for item in items])
    ranked = sorted(range(len(items)), key=lambda i: -scores[i])
    # Recent explicit preferences remain useful even without word overlap.
    return [items[i] for i in ranked[:limit]]


def build_history(messages, turns, budget):
    """Keep complete recent turns and a bounded extractive recap, never hidden reasoning."""
    recent = messages[-turns * 2 :]
    while recent and sum(len(m["content"]) for m in recent) > budget:
        recent = recent[2:]
    earlier = messages[: len(messages) - len(recent)] if recent else messages
    recap = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:160]}" for m in earlier[-6:]
    )[:1000]
    return [{"role": m["role"], "content": m["content"]} for m in recent], recap


def fit_context(messages, budget):
    """Drop whole old user/assistant pairs or complete tool exchanges, not orphan tool IDs."""
    messages = list(messages)

    def length():
        import json

        return len(json.dumps(messages, ensure_ascii=False))

    while length() > budget:
        # Preserve system + current request. Remove oldest ordinary history first.
        if (
            len(messages) > 3
            and messages[1]["role"] == "user"
            and (messages[2]["role"] == "assistant" and not messages[2].get("tool_calls"))
        ):
            del messages[1:3]
            continue
        start = next((i for i, m in enumerate(messages) if m.get("tool_calls")), None)
        if start is not None:
            end = start + 1
            while end < len(messages) and messages[end]["role"] == "tool":
                end += 1
            del messages[start:end]
            continue
        raise ValueError("上下文超过字符预算，请缩短问题或提高 AW_CONTEXT_CHARS")
    return messages
