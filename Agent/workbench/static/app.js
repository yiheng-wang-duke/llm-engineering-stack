const $ = (selector) => document.querySelector(selector);
const state = { session: null, sessions: [], config: null, controller: null, sources: [], events: 0 };
const titles = { chat: "对话工作台", knowledge: "知识库", memory: "长期记忆", settings: "运行配置" };
let toastTimer;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message; node.className = "toast" + (error ? " error" : ""); node.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => node.hidden = true, 5000);
}
function tokenHeaders() {
  const token = sessionStorage.getItem("agent-token");
  return token ? { Authorization: "Bearer " + token } : {};
}
async function api(path, options = {}) {
  const headers = { ...tokenHeaders(), ...options.headers };
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const response = await fetch("/api" + path, { ...options, headers });
  if (!response.ok) {
    let detail = "请求失败（" + response.status + "）";
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body);
    } catch {}
    if (response.status === 401) switchView("settings");
    throw new Error(detail);
  }
  return response;
}
async function json(path, options) { return (await api(path, options)).json(); }
function switchView(name) {
  document.querySelectorAll(".view").forEach(n => n.hidden = n.id !== "view-" + name);
  document.querySelectorAll("[data-view]").forEach(n => n.classList.toggle("active", n.dataset.view === name));
  $("#view-title").textContent = titles[name];
  $("#sidebar").classList.remove("mobile-open");
}
function showDetail(title, content) {
  $("#detail-title").textContent = title;
  $("#detail-body").replaceChildren(typeof content === "string" ? document.createTextNode(content) : content);
  $("#detail-dialog").showModal();
}
function sourceDetail(source) {
  showDetail((source.label ? "[" + source.label + "] " : "") + source.name,
    (source.page ? "第 " + source.page + " 页 · " : "") + "分块 " + (source.ordinal + 1)
    + "\n\n" + source.content);
}
function sourceCard(source) {
  const button = el("button", "source-card");
  button.append(el("strong", "", (source.label ? "[" + source.label + "] " : "") + source.name));
  button.append(el("small", "", (source.page ? "第 " + source.page + " 页 · " : "")
    + "分块 " + (source.ordinal + 1)));
  button.append(el("p", "", source.content));
  button.addEventListener("click", () => sourceDetail(source));
  return button;
}
function renderSources(sources) {
  state.sources = sources;
  $("#source-count").textContent = sources.length;
  $("#sources").replaceChildren(...sources.map(sourceCard));
  if (!sources.length) $("#sources").append(el("div", "empty-small", "本轮尚无检索来源。"));
}
function inlineText(parent, text, sources) {
  // DOM construction only: model/document content never becomes executable HTML.
  const parts = text.split(/(\[S\d+\]|\*\*[^*]+\*\*|`[^`]+`)/g);
  for (const part of parts) {
    const label = /^\[(S\d+)\]$/.exec(part);
    const source = label && sources.find(s => s.label === label[1]);
    if (source) {
      const button = el("button", "citation", part);
      button.addEventListener("click", () => sourceDetail(source));
      parent.append(button);
    } else if (part.startsWith("**") && part.endsWith("**")) {
      parent.append(el("strong", "", part.slice(2, -2)));
    } else if (part.startsWith("`") && part.endsWith("`")) {
      parent.append(el("code", "", part.slice(1, -1)));
    } else parent.append(document.createTextNode(part));
  }
}
function renderAnswer(node, text, sources = []) {
  node.replaceChildren();
  const blocks = text.split("```");
  blocks.forEach((block, index) => {
    if (index % 2) {
      const code = el("code", "", block.replace(/^[a-zA-Z0-9_-]*\n/, ""));
      const pre = el("pre"); pre.append(code); node.append(pre); return;
    }
    for (const paragraph of block.split(/\n\s*\n/)) {
      if (!paragraph) continue;
      const heading = /^#{1,4}\s+/.test(paragraph);
      const p = el(heading ? "h3" : "p");
      inlineText(p, heading ? paragraph.replace(/^#{1,4}\s+/, "") : paragraph, sources);
      node.append(p);
    }
  });
}
function appendMessage(role, content = "", sources = [], runId) {
  $("#welcome")?.remove();
  const row = el("article", "message " + role);
  row.append(el("span", "avatar", role === "user" ? "你" : "✳"));
  const main = el("div", "message-main");
  main.append(el("div", "message-role", role === "user" ? "你" : "Agent Studio"));
  const body = el("div", "message-body");
  if (role === "user") body.textContent = content;
  else renderAnswer(body, content, sources);
  main.append(body); row.append(main);
  if (sources.length) {
    const refs = el("div", "message-sources");
    for (const source of sources) {
      const button = el("button", "", "[" + source.label + "] " + source.name);
      button.addEventListener("click", () => sourceDetail(source)); refs.append(button);
    }
    main.append(refs);
  }
  if (runId && role === "assistant") {
    const button = el("button", "text-button run-link", "查看本轮执行记录 ↗");
    button.addEventListener("click", () => inspectRun(runId)); main.append(button);
  }
  $("#messages").append(row);
  return { row, main, body };
}
function scrollMessages() {
  const node = $("#messages"); node.scrollTop = node.scrollHeight;
}
function resetTrace() {
  state.events = 0; $("#event-count").textContent = "0"; $("#trace").replaceChildren();
  $("#run-metrics").textContent = ""; $("#run-status").textContent = "运行中"; renderSources([]);
}
const eventNames = {
  run: "开始执行", stage: "执行阶段", memory: "读取记忆", memory_saved: "保存记忆",
  context: "组装上下文", tool_call: "调用工具", tool_result: "工具返回", usage: "Token 使用",
  done: "回答完成", error: "执行失败", sources: "知识检索",
};
function traceEvent(kind, data) {
  if (!eventNames[kind]) return;
  state.events++; $("#event-count").textContent = state.events;
  const item = el("div", "trace-event" + (kind === "error" ? " error" : ""));
  const details = el("details");
  const extra = data.name || data.message || (kind === "sources" ? (data.items.length + " 条来源") : "");
  details.append(el("summary", "", eventNames[kind] + (extra ? " · " + extra : "")));
  details.append(el("pre", "", JSON.stringify(data, null, 2))); item.append(details);
  $("#trace").append(item);
}
async function inspectRun(id) {
  try {
    const run = await json("/runs/" + id);
    resetTrace();
    for (const event of run.events) {
      traceEvent(event.kind, event.data);
      if (event.kind === "sources") renderSources(event.data.items);
    }
    $("#run-status").textContent = run.status;
    $("#run-metrics").textContent = "运行编号 " + run.id.slice(0, 12);
    $("#trace-panel").hidden = false;
    if (window.innerWidth <= 1000) $("#trace-panel").classList.add("mobile-open");
  } catch (e) { toast(e.message, true); }
}
async function refreshSessions() {
  state.sessions = await json("/sessions");
  $("#session-count").textContent = state.sessions.length;
  $("#sessions").replaceChildren();
  for (const session of state.sessions) {
    const row = el("div", "session-row" + (session.id === state.session ? " active" : ""));
    const select = el("button", "session-select", session.title);
    select.title = session.title;
    select.addEventListener("click", () => openSession(session.id));
    const remove = el("button", "session-remove", "×");
    remove.setAttribute("aria-label", "删除对话 " + session.title);
    remove.addEventListener("click", async () => {
      if (state.controller || !confirm("删除这段对话及执行记录？")) return;
      try {
        await json("/sessions/" + session.id, { method: "DELETE" });
        if (state.session === session.id) { state.session = null; restoreWelcome(); }
        await refreshSessions();
      } catch (e) { toast(e.message, true); }
    });
    row.append(select, remove); $("#sessions").append(row);
  }
  const selected = state.sessions.find(s => s.id === state.session);
  $("#chat-title").textContent = selected?.title || "新对话";
}
const welcomeTemplate = $("#welcome").cloneNode(true);
function bindSuggestions() {
  document.querySelectorAll("[data-prompt]").forEach(button =>
    button.addEventListener("click", () => { $("#prompt").value = button.dataset.prompt; $("#prompt").focus(); }));
}
function restoreWelcome() {
  $("#messages").replaceChildren(welcomeTemplate.cloneNode(true));
  $("#chat-title").textContent = "新对话"; bindSuggestions();
}
async function openSession(id) {
  if (state.controller) return toast("请先停止当前生成");
  try {
    const data = await json("/sessions/" + id);
    state.session = id; switchView("chat"); $("#messages").replaceChildren();
    if (!data.messages.length) restoreWelcome();
    data.messages.forEach(m => appendMessage(m.role, m.content, m.sources, m.run_id));
    const failed = data.runs.filter(r => r.status !== "completed");
    for (const run of failed.slice(0, 3)) {
      const button = el("button", "text-button", "查看未完成请求：" + run.question.slice(0, 30));
      button.addEventListener("click", () => inspectRun(run.id)); $("#messages").append(button);
    }
    await refreshSessions(); scrollMessages();
    if (data.runs.length) await inspectRun(data.runs[0].id);
  } catch (e) { toast(e.message, true); }
}
async function newChat() {
  if (state.controller) return toast("请先停止当前生成");
  state.session = null; restoreWelcome(); switchView("chat");
  $("#trace-panel").classList.remove("mobile-open");
  resetTrace(); $("#run-status").textContent = "就绪";
  await refreshSessions(); $("#prompt").focus();
}
function setBusy(value) {
  $("#send-button").hidden = value; $("#stop-button").hidden = !value;
  $("#prompt").disabled = value;
  $("#new-chat").disabled = value;
}
async function readEvents(response, onEvent) {
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "";
  const dispatch = frame => {
    let kind = "message", data = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) kind = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (data.length) onEvent(kind, JSON.parse(data.join("\n")));
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        dispatch(buffer.slice(0, boundary)); buffer = buffer.slice(boundary + 2);
      }
      if (done) { if (buffer.trim()) dispatch(buffer); break; }
    }
  } finally { reader.releaseLock(); }
}
$("#chat-form").addEventListener("submit", async event => {
  event.preventDefault();
  const message = $("#prompt").value.trim();
  if (!message || state.controller) return;
  state.controller = new AbortController(); setBusy(true); resetTrace();
  let assistant, answer = "", sources = [], complete = false, runId;
  try {
    if (!state.session) {
      const session = await json("/sessions", { method: "POST", body: JSON.stringify({}) });
      state.session = session.id;
    }
    appendMessage("user", message); assistant = appendMessage("assistant");
    assistant.body.textContent = "正在准备上下文…"; assistant.body.classList.add("pending");
    $("#prompt").value = ""; scrollMessages();
    const response = await api("/sessions/" + state.session + "/chat", {
      method: "POST", signal: state.controller.signal,
      body: JSON.stringify({ message, use_rag: $("#use-rag").checked, use_memory: $("#use-memory").checked }),
    });
    await readEvents(response, (kind, data) => {
      if (kind === "run") runId = data.id;
      if (kind === "response_start") { answer = ""; assistant.body.textContent = "正在生成…"; }
      if (kind === "delta") {
        answer += data.text; assistant.body.classList.remove("pending");
        assistant.body.textContent = answer; scrollMessages();
      }
      if (kind === "sources") { sources = data.items; renderSources(sources); }
      if (kind === "done") {
        complete = true; answer = data.answer; sources = data.sources;
        renderAnswer(assistant.body, answer, sources);
        assistant.body.classList.remove("pending"); $("#run-status").textContent = "完成";
        $("#run-metrics").textContent = "耗时 " + (data.elapsed_ms / 1000).toFixed(2)
          + " s · " + sources.length + " 条来源";
      }
      if (kind === "error") throw new Error(data.message);
      traceEvent(kind, kind === "done" ? { elapsed_ms: data.elapsed_ms } : data);
    });
    if (!complete) throw new Error("连接提前结束，本轮未保存，请重试");
    if (runId) {
      const button = el("button", "text-button run-link", "查看本轮执行记录 ↗");
      button.addEventListener("click", () => inspectRun(runId)); assistant.main.append(button);
    }
  } catch (e) {
    const stopped = e.name === "AbortError";
    const messageText = stopped ? "已停止生成；未完成的回答不会写入历史。" : e.message;
    if (assistant) assistant.main.append(el("div", "message-error", messageText));
    else toast(messageText, !stopped);
    $("#run-status").textContent = stopped ? "已停止" : "失败";
    traceEvent("error", { message: messageText, run_id: runId });
    if (!complete) $("#prompt").value = message;
  } finally {
    state.controller = null; setBusy(false); $("#prompt").focus();
    await Promise.allSettled([refreshSessions(), refreshMemories()]);
  }
});
$("#stop-button").addEventListener("click", () => state.controller?.abort());
$("#prompt").addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); $("#chat-form").requestSubmit();
  }
});
async function refreshDocuments() {
  const docs = await json("/documents");
  $("#doc-count").textContent = docs.length; $("#doc-total").textContent = docs.length;
  $("#document-list").replaceChildren();
  if (!docs.length) $("#document-list").append(el("div", "card-empty", "知识库还是空的。上传第一份文档，或导入示例开始体验。"));
  for (const doc of docs) {
    const card = el("div", "item-card"); card.append(el("span", "item-icon", "▤"));
    const info = el("div", "item-info");
    info.append(el("strong", "", doc.name), el("small", "", doc.chunks + " 个分块 · "
      + (doc.embedding_key === "bm25" ? "BM25" : "向量 + BM25") + " · " + new Date(doc.created_at).toLocaleDateString()));
    const view = el("button", "text-button", "查看");
    view.addEventListener("click", async () => {
      try {
        const data = await json("/documents/" + doc.id);
        showDetail(doc.name, data.chunks.map(c => "── 分块 " + (c.ordinal + 1)
          + (c.page ? " / 第 " + c.page + " 页" : "") + " ──\n" + c.content).join("\n\n"));
      } catch (e) { toast(e.message, true); }
    });
    const remove = el("button", "text-button danger", "删除");
    remove.addEventListener("click", async () => {
      if (!confirm("删除文档及其检索索引？")) return;
      try { await json("/documents/" + doc.id, { method: "DELETE" }); await refreshDocuments(); }
      catch (e) { toast(e.message, true); }
    });
    card.append(info, view, remove); $("#document-list").append(card);
  }
}
async function uploadFiles(files) {
  if (!files.length) return;
  const input = $("#file-input"); input.disabled = true;
  try {
    for (const file of files) {
      $("#upload-hint").textContent = "正在导入 " + file.name + "…";
      const form = new FormData(); form.append("file", file);
      const result = await json("/documents", { method: "POST", body: form });
      toast(result.duplicate ? file.name + " 已存在" : file.name + " 已索引");
    }
  } catch (e) { toast(e.message, true); }
  finally {
    input.disabled = false; input.value = "";
    $("#upload-hint").textContent = "支持 PDF、Markdown、TXT、CSV、JSON";
    await refreshDocuments().catch(e => toast(e.message, true));
  }
}
$("#file-input").addEventListener("change", event => uploadFiles([...event.target.files]));
const dropZone = $("#drop-zone");
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag");
  if (!$("#file-input").disabled) uploadFiles([...e.dataTransfer.files]);
});
$("#load-example").addEventListener("click", async () => {
  try {
    const response = await fetch("/static/example.md");
    if (!response.ok) throw new Error("示例文件不可用");
    await uploadFiles([new File([await response.text()], "项目手册.md", { type: "text/markdown" })]);
  } catch (e) { toast(e.message, true); }
});
$("#search-form").addEventListener("submit", async e => {
  e.preventDefault(); const button = e.currentTarget.querySelector("button"); button.disabled = true;
  try {
    const results = await json("/knowledge/search", { method: "POST",
      body: JSON.stringify({ query: $("#search-query").value }) });
    $("#search-results").replaceChildren(...results.map(sourceCard));
    if (!results.length) $("#search-results").append(el("div", "empty-small", "没有检索结果。试试其他关键词或添加相关文档。"));
  } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
});
$("#reindex").addEventListener("click", async e => {
  e.currentTarget.disabled = true;
  try { await json("/knowledge/reindex", { method: "POST" }); toast("索引重建完成"); await refreshDocuments(); }
  catch (error) { toast(error.message, true); } finally { $("#reindex").disabled = false; }
});
async function refreshMemories() {
  const memories = await json("/memories");
  $("#memory-count").textContent = memories.length; $("#memory-total").textContent = memories.length;
  $("#memory-list").replaceChildren();
  if (!memories.length) $("#memory-list").append(el("div", "card-empty", "还没有长期记忆。保存你的偏好或项目背景，让后续对话更连贯。"));
  for (const memory of memories) {
    const card = el("div", "item-card"); card.append(el("span", "item-icon", "◎"));
    const info = el("div", "item-info");
    info.append(el("strong", "", memory.content), el("small", "", "用户主动保存 · "
      + new Date(memory.created_at).toLocaleString()));
    const remove = el("button", "text-button danger", "删除");
    remove.addEventListener("click", async () => {
      if (!confirm("删除这条长期记忆？")) return;
      try { await json("/memories/" + memory.id, { method: "DELETE" }); await refreshMemories(); }
      catch (e) { toast(e.message, true); }
    });
    card.append(info, remove); $("#memory-list").append(card);
  }
}
$("#memory-form").addEventListener("submit", async e => {
  e.preventDefault(); const button = e.currentTarget.querySelector("button"); button.disabled = true;
  try {
    await json("/memories", { method: "POST", body: JSON.stringify({ content: $("#memory-content").value }) });
    $("#memory-content").value = ""; await refreshMemories(); toast("记忆已保存");
  } catch (error) { toast(error.message, true); } finally { button.disabled = false; }
});
async function loadConfig() {
  state.config = await json("/config"); const c = state.config;
  $("#mode-badge").textContent = c.mode === "demo" ? "离线演示 · 无模型" : "vLLM";
  $("#mode-badge").classList.toggle("demo", c.mode === "demo");
  $("#model-label").textContent = c.mode === "demo" ? "DEMO · 未调用模型" : c.model;
  $("#banner").hidden = c.mode !== "demo";
  $("#banner").textContent = "当前为离线演示：回答来自固定演示逻辑。文档、检索、记忆和历史存储正常工作；连接 vLLM 后启用真实 Agent。";
  $("#runtime-config").replaceChildren();
  for (const [label, value] of [
    ["推理模式", c.mode === "demo" ? "离线演示（无模型推理）" : "vLLM · OpenAI 兼容接口"],
    ["聊天模型", c.model], ["检索引擎", c.embedding_backend === "vllm" ? "Qwen Embedding + BM25 / RRF" : "BM25 · 中英文关键词检索"],
    ["向量模型", c.embedding_model || "未启用"], ["执行预算", c.max_steps + " 轮（含最终回答）"],
    ["持久化存储", "SQLite · 本地磁盘"],
  ]) {
    const card = el("div", "config-card"); card.append(el("small", "", label), el("strong", "", value));
    $("#runtime-config").append(card);
  }
}
async function checkHealth() {
  $("#health-label").textContent = "检查中";
  try {
    const h = await json("/health"); const ready = h.inference.ready && h.embedding.ready;
    $("#health-label").textContent = state.config?.mode === "demo" ? "演示模式" : ready ? "服务已连接" : "服务未就绪";
    $("#health-button").classList.toggle("offline", !ready);
    $("#health-button").title = [h.inference.detail, h.embedding.detail].filter(Boolean).join("；");
    if (state.config?.mode !== "demo") {
      $("#banner").hidden = ready;
      $("#banner").textContent = "推理或向量服务未就绪。请在运行配置中确认模型，并按 README 启动 vLLM。";
    }
  } catch (e) { $("#health-label").textContent = "连接失败"; toast(e.message, true); }
}
$("#token-form").addEventListener("submit", async e => {
  e.preventDefault(); sessionStorage.setItem("agent-token", $("#api-token").value.trim());
  await initialize();
});
$("#api-token").value = sessionStorage.getItem("agent-token") || "";
$("#health-button").addEventListener("click", checkHealth);
$("#new-chat").addEventListener("click", () => newChat().catch(e => toast(e.message, true)));
$("#toggle-trace").addEventListener("click", () => {
  const panel = $("#trace-panel");
  if (window.innerWidth <= 1000) { panel.hidden = false; panel.classList.toggle("mobile-open"); }
  else panel.hidden = !panel.hidden;
});
$("#close-trace").addEventListener("click", () => $("#trace-panel").classList.remove("mobile-open"));
$("#menu-button").addEventListener("click", () => $("#sidebar").classList.toggle("mobile-open"));
$("#close-dialog").addEventListener("click", () => $("#detail-dialog").close());
$("#detail-dialog").addEventListener("click", e => { if (e.target === $("#detail-dialog")) $("#detail-dialog").close(); });
$("#export-chat").addEventListener("click", async () => {
  if (!state.session) return toast("当前还没有对话");
  try {
    const response = await api("/sessions/" + state.session + "/export");
    const url = URL.createObjectURL(await response.blob()); const link = el("a");
    link.href = url; link.download = "agent-chat-" + state.session + ".json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { toast(e.message, true); }
});
document.querySelectorAll("[data-view]").forEach(button =>
  button.addEventListener("click", () => switchView(button.dataset.view)));
document.addEventListener("keydown", e => {
  if (e.key.toLowerCase() === "n" && !e.ctrlKey && !e.metaKey && !e.altKey
      && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName) && !$("#detail-dialog").open) {
    newChat().catch(error => toast(error.message, true));
  }
});
bindSuggestions();
async function initialize() {
  try {
    await loadConfig();
    const results = await Promise.allSettled([refreshSessions(), refreshDocuments(), refreshMemories()]);
    for (const result of results) if (result.status === "rejected") toast(result.reason.message, true);
    await checkHealth();
  } catch (e) { toast(e.message, true); }
}
initialize();
