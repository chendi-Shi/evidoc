"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { documents: [], selected: new Set(), status: null, busy: false, asking: false, sources: [], evaluationLoaded: false, view: "chat", trainingState: null, trainingLoading: false };
  const modes = { hybrid: "混合检索", bm25: "关键词 BM25", dense: "语义检索", hybrid_fallback_bm25: "关键词 BM25（回退）" };
  const emptyEvidence = $("evidence-content").cloneNode(true);
  const emptyEvaluation = $("evaluation-content").cloneNode(true);
  let toastTimer;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#i-${name}`);
    svg.setAttribute("aria-hidden", "true");
    svg.append(use);
    return svg;
  }

  function toast(message, isError = false) {
    const region = $("toast-region");
    region.replaceChildren(el("div", `toast${isError ? " error" : ""}`, message));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => region.replaceChildren(), isError ? 9000 : 5500);
  }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeout || 480000);
    try {
      const response = await fetch(path, { ...options, signal: controller.signal });
      const raw = await response.text();
      let body;
      try { body = raw ? JSON.parse(raw) : {}; } catch { throw new Error("本地服务返回了无法读取的内容，请检查服务日志。"); }
      if (!response.ok) {
        let detail = body.detail || body.error || `请求失败（HTTP ${response.status}）`;
        if (Array.isArray(detail)) detail = detail.map((item) => item.msg || JSON.stringify(item)).join("；");
        if (typeof detail !== "string") detail = JSON.stringify(detail);
        throw new Error(detail);
      }
      return body;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("等待本地服务超时。任务可能仍在后台运行，请查看服务日志后重试。");
      if (error instanceof TypeError) throw new Error("暂时无法连接本地服务。请确认 Evidoc 已启动，再刷新页面。");
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function setBusy(busy) {
    state.busy = busy;
    ["send-button", "demo-button", "upload-zone", "upload-shortcut", "reindex-button", "evaluate-button"].forEach((id) => { $(id).disabled = busy; });
    document.querySelectorAll(".delete-document, .document-item input").forEach((button) => { button.disabled = busy; });
    $("reset-button").disabled = state.asking;
    $("question-form").setAttribute("aria-busy", String(state.asking));
  }

  function operationLabel(id, text) {
    const button = $(id);
    const original = [...button.childNodes];
    button.replaceChildren(el("span", "spinner"), document.createTextNode(text));
    return () => button.replaceChildren(...original);
  }

  function displayWarnings(warnings) {
    if (Array.isArray(warnings) && warnings.length) toast(warnings.map((warning) => typeof warning === "string" ? warning : JSON.stringify(warning)).join("；"));
  }

  function modelCaption(status) {
    if (!status) return "尚未连接 · 请确认本地服务已启动";
    if (status.model_ready) return `${status.model_name || "本地模型"} · 依据文档作答`;
    return "证据摘录模式 · 生成模型尚未就绪";
  }

  function renderStatus() {
    const status = state.status;
    $("connection-dot").classList.toggle("offline", !status);
    $("answer-mode-caption").textContent = modelCaption(status);
    if (!status) {
      $("sidebar-device").textContent = "本地服务未连接";
      const box = el("div", "notice");
      box.append(el("strong", "", "等待本地服务"), el("p", "", "请先启动 Evidoc 服务，然后刷新页面。这里会显示实际设备、模型和索引状态。"));
      $("status-content").replaceChildren(box);
      return;
    }
    const hardware = status.hardware || {};
    $("sidebar-device").textContent = hardware.gpu || status.backend || "本地服务已连接";
    const grid = el("div", "status-grid");
    const values = [
      ["计算设备", hardware.gpu || "未报告 GPU", hardware.vram_gb ? `${hardware.vram_gb} GB 显存${hardware.driver ? ` · 驱动 ${hardware.driver}` : ""}` : "设备信息来自本地服务"],
      ["推理后端", status.backend || "未报告", `运行模式：${status.mode || "本地"}`],
      ["生成模型", status.model_name || "未配置", status.model_ready ? "已就绪 · 可依据证据生成答案" : "尚未就绪 · 使用证据摘录回答"],
      ["语义索引", status.embedding_ready ? "已就绪" : "尚未就绪", status.embedding_ready ? "支持向量检索与混合检索" : "关键词检索仍可使用；实际模式以每次返回为准"],
      ["文档资料库", `${status.documents ?? state.documents.length} 份文档`, "导入的资料保存在本地工作目录"],
      ["文本片段", `${status.chunks ?? 0} 个片段`, "回答中的引用可定位到检索原文"]
    ];
    for (const [label, value, description] of values) {
      const tile = el("div", "status-tile");
      tile.append(el("span", "", label), el("strong", "", value), el("small", "", description));
      grid.append(tile);
    }
    $("status-content").replaceChildren(grid);
  }

  function renderDocuments() {
    $("document-count").textContent = String(state.documents.length);
    const list = $("document-list");
    list.replaceChildren();
    if (!state.documents.length) {
      const empty = el("div", "library-empty");
      empty.append(document.createTextNode("一份资料，一个新的起点。"), el("br"), document.createTextNode("先添加文档，或用示例试一试。"));
      list.append(empty);
      $("selection-caption").textContent = "问题的答案，从你的资料开始。";
      return;
    }
    for (const doc of state.documents) {
      const row = el("div", "document-item");
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = state.selected.has(doc.id);
      checkbox.disabled = state.busy;
      checkbox.id = `doc-${doc.id}`;
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) state.selected.add(doc.id); else state.selected.delete(doc.id);
        renderSelection();
      });
      const label = el("label", "document-label");
      label.htmlFor = checkbox.id;
      label.append(el("div", "document-name", doc.name || "未命名文档"), el("div", "document-meta", `${doc.chunks ?? 0} 个片段${doc.pages ? ` · ${doc.pages} 页` : ""}`));
      const remove = el("button", "icon-button delete-document");
      remove.type = "button";
      remove.title = `移除 ${doc.name}`;
      remove.setAttribute("aria-label", `移除文档 ${doc.name}`);
      remove.append(icon("trash"));
      remove.disabled = state.busy;
      remove.addEventListener("click", () => removeDocument(doc));
      row.append(checkbox, label, remove);
      list.append(row);
    }
    renderSelection();
  }

  function renderSelection() {
    $("selection-caption").textContent = state.selected.size === state.documents.length ? `已选全部 ${state.documents.length} 份资料 · 可取消勾选缩小范围` : `已选 ${state.selected.size} / ${state.documents.length} 份资料`;
  }

  async function refresh({ quiet = false } = {}) {
    const results = await Promise.allSettled([api("/api/status", { timeout: 20000 }), api("/api/documents", { timeout: 20000 })]);
    if (results[0].status === "fulfilled") state.status = results[0].value;
    else { state.status = null; if (!quiet) toast(results[0].reason.message, true); }
    if (results[1].status === "fulfilled") {
      const previous = new Set(state.documents.map((doc) => doc.id));
      state.documents = results[1].value.documents || [];
      const valid = new Set(state.documents.map((doc) => doc.id));
      state.selected = new Set([...state.selected].filter((id) => valid.has(id)));
      state.documents.forEach((doc) => { if (!previous.has(doc.id)) state.selected.add(doc.id); });
      renderDocuments();
    } else if (results[0].status === "fulfilled" && !quiet) toast(results[1].reason.message, true);
    renderStatus();
  }

  async function uploadFiles(files) {
    if (state.busy || !files.length) return;
    if (files.length > 10) { toast("每批最多上传 10 个文件，请分批添加。", true); return; }
    const form = new FormData();
    [...files].forEach((file) => form.append("files", file));
    setBusy(true);
    const restore = operationLabel("upload-zone", "正在解析与索引…");
    try {
      const result = await api("/api/documents", { method: "POST", body: form });
      await refresh({ quiet: true });
      toast(`文档处理完成，当前有 ${state.documents.length} 份资料。`);
      displayWarnings(result.warnings);
    } catch (error) { toast(error.message, true); }
    finally { restore(); $("file-input").value = ""; setBusy(false); }
  }

  async function importDemo() {
    if (state.busy) return;
    setBusy(true);
    const restore = operationLabel("demo-button", "正在导入示例…");
    try {
      const result = await api("/api/demo", { method: "POST" });
      await refresh({ quiet: true });
      toast("示例资料已就绪。选择一个问题，开始查看证据。");
      displayWarnings(result.warnings);
    } catch (error) { toast(error.message, true); }
    finally { restore(); setBusy(false); }
  }

  async function removeDocument(doc) {
    if (state.busy) return;
    if (!window.confirm(`从 Evidoc 资料库移除「${doc.name}」？`)) return;
    setBusy(true);
    try {
      await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: "DELETE" });
      await refresh({ quiet: true });
      toast("已从资料库移除文档。");
    } catch (error) { toast(error.message, true); }
    finally { setBusy(false); }
  }

  async function reindex() {
    if (state.busy) return;
    if (!state.documents.length) { toast("请先添加文档，再重建索引。"); return; }
    setBusy(true);
    try {
      toast("正在重建索引，完成后会更新资料库。");
      const result = await api("/api/reindex", { method: "POST" });
      await refresh({ quiet: true });
      toast(`索引重建完成${result.indexed != null ? `：${result.indexed} 个片段` : ""}。`);
      displayWarnings(result.warnings);
    } catch (error) { toast(error.message, true); }
    finally { setBusy(false); }
  }

  function switchView(name) {
    const titles = { chat: "文档问答", training: "训练实验台", evaluation: "检索实验室", overview: "本地运行状态" };
    state.view = name;
    document.querySelectorAll(".nav-button").forEach((button) => {
      const active = button.dataset.view === name;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    });
    Object.keys(titles).forEach((view) => { $(`${view}-view`).hidden = view !== name; });
    $("view-title").textContent = titles[name];
    if (name === "overview") refresh({ quiet: true });
    if (name === "evaluation" && !state.evaluationLoaded) loadEvaluation();
    if (name === "training") loadTraining();
  }

  function scrollChat() {
    const scroller = $("chat-scroll");
    scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
  }

  function addMessage(role, text) {
    $("welcome").hidden = true;
    const message = el("article", `message ${role}`);
    const heading = el("div", "message-role");
    heading.append(el("span", "message-avatar", role === "user" ? "你" : "见"), document.createTextNode(role === "user" ? "你的问题" : "见微 · 文档助手"));
    const body = el("div", "message-body", text);
    message.append(heading, body);
    $("messages").append(message);
    scrollChat();
    return { message, body };
  }

  function showSources(sources) {
    state.sources = sources;
    const content = $("evidence-content");
    $("evidence-count").textContent = String(sources.length);
    content.replaceChildren();
    if (!sources.length) {
      const empty = el("div", "evidence-empty");
      empty.append(el("h3", "", "未找到可展示的证据"), el("p", "", "试着具体描述问题，或添加相关资料。"));
      content.append(empty);
      return;
    }
    content.append(el("div", "source-explanation", "以下为本次检索的原文片段。检索分数用于排序，不代表答案正确率。"));
    sources.forEach((source, index) => {
      const citation = source.citation ?? index + 1;
      const card = el("article", "source-card");
      card.id = `source-${citation}`;
      card.tabIndex = -1;
      card.setAttribute("aria-label", `引用 ${citation}，${source.name || "文档"}`);
      const heading = el("div", "source-card-head");
      heading.append(el("span", "source-number", `[${citation}]`), el("span", "source-name", source.name || "未命名文档"));
      const meta = el("div", "source-meta");
      meta.append(el("span", "", source.page != null ? `第 ${source.page} 页` : "文本片段"), el("span", "", Number.isFinite(Number(source.score)) ? `检索分数 ${Number(source.score).toFixed(3)}` : "原文引用"));
      card.append(heading, el("div", "source-text", source.text || ""), meta);
      content.append(card);
    });
  }

  function focusCitation(citation, sources) {
    showSources(sources);
    const card = $(`source-${citation}`);
    if (!card) { toast("这条引用没有对应的原文片段。"); return; }
    card.focus({ preventScroll: true });
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    card.classList.add("highlight");
    setTimeout(() => card.classList.remove("highlight"), 1800);
  }

  function renderAnswer(body, text, sources) {
    body.replaceChildren();
    const citations = new Set(sources.map((source, index) => Number(source.citation ?? index + 1)));
    const parts = String(text || "本次没有生成回答，请查看原文证据。").split(/(\[\d+\])/g);
    for (const part of parts) {
      const match = /^\[(\d+)\]$/.exec(part);
      if (match && citations.has(Number(match[1]))) {
        const button = el("button", "citation-button", part);
        button.type = "button";
        button.setAttribute("aria-label", `查看引用 ${match[1]} 的原文`);
        button.addEventListener("click", () => focusCitation(Number(match[1]), sources));
        body.append(button);
      } else body.append(document.createTextNode(part));
    }
  }

  function renderTrace(message, result) {
    const timings = result.timings || {};
    const details = el("details", "trace");
    const total = Number.isFinite(Number(timings.total_ms)) ? ` · ${(Number(timings.total_ms) / 1000).toFixed(2)}s` : "";
    details.append(el("summary", "", `处理过程 · ${modes[result.mode_used] || result.mode_used || "未报告检索模式"}${total}`));
    const list = el("ol", "trace-list");
    const steps = Array.isArray(result.trace) ? result.trace : [];
    for (const item of steps) {
      const li = el("li");
      const description = el("div");
      description.append(el("strong", "", item.step || "处理步骤"), el("p", "", item.detail || ""));
      li.append(description, el("span", "mono", Number.isFinite(Number(item.ms)) ? `${Math.round(Number(item.ms))}ms` : ""));
      list.append(li);
    }
    if (!steps.length) {
      for (const [label, key] of [["证据检索", "retrieval_ms"], ["回答生成", "generation_ms"]]) {
        if (timings[key] != null) { const item = el("li"); item.append(el("strong", "", label), el("span", "mono", `${Math.round(Number(timings[key]))}ms`)); list.append(item); }
      }
    }
    details.append(list);
    message.append(details);
  }

  async function ask(event) {
    event.preventDefault();
    if (state.busy) return;
    const question = $("question").value.trim();
    if (!question) { $("question").focus(); return; }
    if (!state.documents.length) { toast("先添加文档，或点击「导入示例资料」，就可以开始提问。"); return; }
    if (!state.selected.size) { toast("请先在左侧勾选至少一份资料。"); return; }
    const payload = { question, mode: $("retrieval-mode").value, top_k: Number($("top-k").value), document_ids: state.selected.size === state.documents.length ? [] : [...state.selected] };
    addMessage("user", question);
    state.sources = [];
    $("evidence-count").textContent = "0";
    $("evidence-content").replaceChildren(el("p", "source-explanation", "正在等待本次检索结果，证据会随回答一起显示。"));
    $("question").value = "";
    $("question").style.height = "";
    state.asking = true;
    setBusy(true);
    $("pending-status").hidden = false;
    $("pending-label").textContent = "已提交 · 正在检索与作答";
    $("pending-time").textContent = "0s";
    const started = performance.now();
    const interval = setInterval(() => {
      const seconds = Math.floor((performance.now() - started) / 1000);
      $("pending-time").textContent = `${seconds}s`;
      if (seconds >= 25) $("pending-label").textContent = "本地任务处理中 · 首次加载模型可能较慢";
      if (seconds >= 90) $("pending-label").textContent = "仍在等待本地服务 · 可查看推理服务日志";
    }, 1000);
    try {
      const result = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const sources = Array.isArray(result.sources) ? result.sources : [];
      const { message, body } = addMessage("assistant");
      renderAnswer(body, result.answer, sources);
      showSources(sources);
      const captions = [];
      if (result.answer_mode === "extractive") captions.push("证据摘录模式：以下内容来自检索原文，未使用生成模型改写。");
      else if (result.answer_mode === "refused") captions.push("当前资料不足以支持回答。可以补充相关文档，或换一种更具体的问法。");
      else if (result.answer_mode === "generated") captions.push(`由 ${result.model || "本地模型"} 根据证据生成，请结合引用核验。`);
      else if (result.answer_mode === "finetuned") captions.push(`由 ${result.model || "本地 LoRA 模型"} 提取答案；已核对逐字出处，请自行核验是否回答了问题。`);
      if (result.citation_valid === false && result.answer_mode !== "refused") captions.push("引用未通过出处校验，请核验原文。");
      if (Array.isArray(result.warnings)) captions.push(...result.warnings.map(String));
      if (captions.length) message.append(el("p", "answer-note", captions.join(" ")));
      renderTrace(message, result);
      scrollChat();
    } catch (error) {
      showSources([]);
      const { message, body } = addMessage("assistant");
      body.className = "message-error";
      body.textContent = error.message;
      const retry = el("button", "text-button", "重新填写这个问题 ↗");
      retry.type = "button";
      retry.addEventListener("click", () => { $("question").value = question; $("question").focus(); });
      message.append(retry);
      scrollChat();
    } finally {
      clearInterval(interval);
      $("pending-status").hidden = true;
      state.asking = false;
      setBusy(false);
      refresh({ quiet: true });
    }
  }

  function resetChat() {
    if (state.asking) return;
    $("messages").replaceChildren();
    $("welcome").hidden = false;
    state.sources = [];
    $("evidence-count").textContent = "0";
    $("evidence-content").replaceChildren(...[...emptyEvidence.childNodes].map((child) => child.cloneNode(true)));
    $("question").value = "";
    $("question").focus();
  }

  function percent(value) { return Number.isFinite(Number(value)) && value != null ? `${(Number(value) * 100).toFixed(1)}%` : "—"; }

  function renderEvaluation(report) {
    const content = $("evaluation-content");
    content.replaceChildren();
    if (!report.available) { content.append(...[...emptyEvaluation.childNodes].map((node) => node.cloneNode(true))); return; }
    const wrap = el("div", "evaluation-table-wrap");
    const table = el("table", "evaluation-table");
    const head = el("thead");
    const headRow = el("tr");
    ["检索策略", "Recall@5", "MRR@5", "无答案空召回率", "平均耗时"].forEach((heading) => { const cell = el("th", "", heading); cell.scope = "col"; headRow.append(cell); });
    head.append(headRow);
    const tbody = el("tbody");
    for (const row of report.results || []) {
      const tr = el("tr");
      const label = modes[row.mode] || row.mode || "未报告";
      const modeUsed = row.mode_used && row.mode_used !== row.mode ? ` → ${modes[row.mode_used] || row.mode_used}` : "";
      [label + modeUsed, percent(row.recall_at_5), percent(row.mrr_at_5), percent(row.unanswerable_accuracy), row.latency_ms != null ? `${Number(row.latency_ms).toFixed(1)} ms` : "—"].forEach((value) => tr.append(el("td", "", value)));
      tbody.append(tr);
    }
    table.append(head, tbody);
    wrap.append(table);
    const time = report.created_at ? new Date(report.created_at) : null;
    content.append(wrap, el("p", "report-caption", `${report.dataset || "内置合成测试集"} · ${report.sample_count ?? "—"} 个问题${time && !Number.isNaN(time.valueOf()) ? ` · ${time.toLocaleString("zh-CN")}` : ""}`));
    (report.notes || []).forEach((note) => content.append(el("p", "report-note", note)));
  }

  async function loadEvaluation() {
    try { const report = await api("/api/evaluation", { timeout: 20000 }); renderEvaluation(report); state.evaluationLoaded = true; }
    catch (error) { toast(error.message, true); }
  }

  async function evaluate() {
    if (state.busy) return;
    setBusy(true);
    const restore = operationLabel("evaluate-button", "评测运行中…");
    try {
      const report = await api("/api/evaluate", { method: "POST" });
      renderEvaluation(report);
      state.evaluationLoaded = true;
      await refresh({ quiet: true });
      toast("评测已完成，以下数据来自本次实际运行。");
    } catch (error) { toast(error.message, true); }
    finally { restore(); setBusy(false); }
  }

  function trainingNumber(value, suffix = "", decimals = 0) {
    return value != null && Number.isFinite(Number(value)) ? `${Number(value).toFixed(decimals)}${suffix}` : "—";
  }

  function renderTraining(report) {
    const content = $("training-content");
    content.replaceChildren();
    state.trainingState = report.state;
    const names = { not_started: "尚未开始", preparing: "准备数据与模型", running: "正在训练", completed: "训练已完成", published: "已发布的实验结果", failed: "训练中断" };
    const summary = el("div", "training-summary");
    const badge = el("span", `training-state ${report.state === "failed" ? "failed" : ""}`, names[report.state] || "状态未知");
    summary.append(badge, el("p", "", report.message || (report.state === "not_started" ? "等待 GPU 小规模试跑，训练由本地命令行启动。" : "以下数据读取自本地训练记录。")));
    const device = typeof report.device === "string" ? report.device : "";
    const verifiedCuda = report.gpu_verified === true || device.toLowerCase().startsWith("cuda");
    const gpuName = report.gpu_name || report.gpu || state.status?.hardware?.gpu || "GPU";
    if (verifiedCuda) {
      const gpu = el("div", "training-gpu-badge");
      gpu.append(icon("cpu"), el("strong", "", `CUDA · ${gpuName}`), el("span", "", report.state === "running" ? "正在进行 GPU 训练" : "GPU 设备已验证"));
      summary.append(gpu);
    }
    content.append(summary);
    if (report.state === "published") return;
    const grid = el("div", "status-grid");
    const values = [
      ["训练模型", report.model || "尚未配置", report.quantization || "量化配置尚未报告"],
      ["训练设备", verifiedCuda ? device || "CUDA" : device || "尚未验证", verifiedCuda ? gpuName : "等待训练器报告实际 CUDA 设备"],
      ["已完成步数", trainingNumber(report.steps), "来自训练器的实际进度"],
      ["最近训练损失", trainingNumber(report.loss, "", 4), "Loss · 仅用于观察优化过程"],
      ["运行时间", trainingNumber(report.elapsed_hours, " 小时", 2), `时间预算：${trainingNumber(report.budget_hours, " 小时", 1)}`],
      ["显存峰值", trainingNumber(report.peak_vram_gb, " GB", 2), "来自训练器运行记录"],
      ["产出文件", `${Array.isArray(report.artifacts) ? report.artifacts.length : 0} 项`, "模型、适配器与实验记录"],
      ["记录来源", "本地训练器", "仅展示实际写入的状态与指标"]
    ];
    values.forEach(([label, value, description]) => {
      const tile = el("div", "status-tile");
      tile.append(el("span", "", label), el("strong", "", value), el("small", "", description));
      grid.append(tile);
    });
    content.append(grid);
    const history = (Array.isArray(report.history) ? report.history : []).filter((point) => Number.isFinite(Number(point.step)) && Number.isFinite(Number(point.loss)) && point.loss != null && point.step != null).sort((a, b) => Number(a.step) - Number(b.step));
    if (history.length >= 2) {
      const chart = el("div", "training-chart");
      chart.append(el("h3", "", "训练损失 · 实际记录"));
      const chartSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      chartSvg.setAttribute("viewBox", "0 0 500 180");
      chartSvg.setAttribute("role", "img");
      chartSvg.setAttribute("aria-label", `训练损失曲线，${history.length} 个记录点，从 ${history[0].loss} 到 ${history[history.length - 1].loss}`);
      const minStep = Number(history[0].step), maxStep = Number(history[history.length - 1].step);
      const losses = history.map((point) => Number(point.loss));
      const minLoss = Math.min(...losses), maxLoss = Math.max(...losses);
      const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      line.setAttribute("points", history.map((point) => `${12 + (Number(point.step) - minStep) / (maxStep - minStep || 1) * 476},${164 - (Number(point.loss) - minLoss) / (maxLoss - minLoss || 1) * 150}`).join(" "));
      line.setAttribute("fill", "none"); line.setAttribute("stroke", "#58844c"); line.setAttribute("stroke-width", "2");
      chartSvg.append(line);
      const axis = el("div", "training-chart-axis");
      axis.append(el("span", "mono", `STEP ${minStep}`), el("span", "mono", `LOSS ${minLoss.toFixed(3)}–${maxLoss.toFixed(3)}`), el("span", "mono", `STEP ${maxStep}`));
      chart.append(chartSvg, axis);
      content.append(chart);
    } else content.append(el("p", "report-caption", "有至少两个有效训练记录点后，将显示实际损失曲线。"));
    if (Array.isArray(report.artifacts) && report.artifacts.length) {
      const artifacts = el("div", "training-artifacts");
      artifacts.append(el("h3", "", "本地产出"));
      report.artifacts.forEach((artifact) => artifacts.append(el("div", "training-artifact", typeof artifact === "string" ? artifact : artifact.path || artifact.name || JSON.stringify(artifact))));
      content.append(artifacts);
    }
  }

  async function loadTraining() {
    if (state.trainingLoading) return;
    state.trainingLoading = true;
    $("training-refresh-button").disabled = true;
    try {
      const [training, experiments] = await Promise.all([
        api("/api/training", { timeout: 20000 }), api("/api/experiments", { timeout: 20000 }),
      ]);
      renderTraining(training);
      renderExperiments(experiments);
    }
    catch (error) {
      const empty = el("div", "empty-result");
      empty.append(el("span", "empty-symbol", "⌘"), el("h3", "", "暂未读取到训练记录"), el("p", "", "训练服务未就绪或记录尚未生成。请完成 GPU 试跑后刷新状态。"));
      $("training-content").replaceChildren(empty);
      state.trainingState = null;
    } finally { state.trainingLoading = false; $("training-refresh-button").disabled = false; }
  }

  function renderExperiments(report) {
    const content = $("experiment-results");
    content.replaceChildren();
    if (!report.available || !report.aggregate) return;
    content.append(el("h2", "", "已完成的保留集实验"));
    const note = el("p", "report-caption", "同一组 200 道留出题 · 每配置 3 个随机种子 · 均值 ± 样本标准差。自定义 EM / 字符 F1，不是官方 CMRC 分数；空上下文探针不代表真实无答案能力。");
    content.append(note);
    const table = el("table", "experiment-table");
    const head = el("tr");
    ["配置", "完全匹配 EM", "字符 F1"].forEach((name) => head.append(el("th", "", name)));
    const thead = el("thead"); thead.append(head); table.append(thead);
    const body = el("tbody");
    const format = (metric) => metric && Number.isFinite(metric.mean) ? `${(metric.mean * 100).toFixed(2)}% ± ${(metric.sample_stdev * 100).toFixed(2)}` : "—";
    const base = report.runs && report.runs[0] && report.runs[0].results.base;
    if (base) {
      const row = el("tr");
      ["基础模型", `${(base.normalized_em * 100).toFixed(2)}%`, `${(base.character_f1 * 100).toFixed(2)}%`].forEach((text) => row.append(el("td", "", text)));
      body.append(row);
    }
    const labels = { "r8-auto": "LoRA r8 · 混合投影", "r16-auto": "LoRA r16 · 混合投影", "r8-q_proj,v_proj": "LoRA r8 · 全注意力投影" };
    Object.entries(report.aggregate).forEach(([key, result]) => {
      const row = el("tr");
      [labels[key] || key, format(result.normalized_em), format(result.character_f1)].forEach((text) => row.append(el("td", "", text)));
      body.append(row);
    });
    table.append(body); content.append(table);
  }

  document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => {
    $("question").value = button.dataset.question;
    $("question").focus();
    if (!state.documents.length) toast("先点击左侧「导入示例资料」，再发送这个问题。");
  }));
  $("upload-zone").addEventListener("click", () => $("file-input").click());
  $("upload-shortcut").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (event) => uploadFiles(event.target.files));
  $("upload-zone").addEventListener("dragover", (event) => { event.preventDefault(); if (!state.busy) $("upload-zone").classList.add("dragging"); });
  $("upload-zone").addEventListener("dragleave", () => $("upload-zone").classList.remove("dragging"));
  $("upload-zone").addEventListener("drop", (event) => { event.preventDefault(); $("upload-zone").classList.remove("dragging"); uploadFiles(event.dataTransfer.files); });
  $("demo-button").addEventListener("click", importDemo);
  $("reindex-button").addEventListener("click", reindex);
  $("evaluate-button").addEventListener("click", evaluate);
  $("training-refresh-button").addEventListener("click", loadTraining);
  $("reset-button").addEventListener("click", resetChat);
  $("question-form").addEventListener("submit", ask);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("question-form").requestSubmit(); }
  });
  $("question").addEventListener("input", () => { $("question").style.height = "auto"; $("question").style.height = `${Math.min(155, $("question").scrollHeight)}px`; });
  refresh();
  setInterval(() => { if (state.view === "training" && ["running", "preparing"].includes(state.trainingState) && !document.hidden) loadTraining(); }, 15000);
})();
