// 前端冒烟：把 index.html 里的内联 <script> 拿出来，配一个迷你 DOM 跑一遍。
//
// 为什么会有这个文件（两次真实事故换来的）：
//   1. T37：进度条压根没渲染——`<li>` 只塞进了 Map，没 appendChild 到 DOM。
//      后端的 SSE 事件序列全是绿的，坏的是"画出来"这一步，谁都没测；
//   2. T39：改 `stepLine` 函数签名时漏改了调用处，`event` 变成 undefined，
//      界面上是"连接中断：TypeError: Cannot read properties of undefined (reading 'stage')"。
//
// 两次都是"后端协议正确、DOM 写入错误"，而这恰好是能用迷你 DOM 测出来的：
// 不需要浏览器、不需要 jsdom，一个 makeEl() 就够。
//
// 跑法：node tests/frontend_smoke.mjs（pytest 里由 test_web_frontend.py 代跑，没 node 就 skip）

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../web/static/index.html", import.meta.url), "utf8");
const matched = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(matched, "index.html 里找不到内联 <script>（结构变了？这个测试要跟着改）");
const src = matched[1];

// ---- 迷你 DOM：够用就好，不追求像浏览器 ----
function makeEl(tag = "div") {
  const node = {
    tagName: tag,
    className: "",
    textContent: "",
    value: "",
    placeholder: "",
    disabled: false,
    maxLength: 0,
    innerHTML: "",
    style: {},
    dataset: {},
    children: [],
    handlers: {},
  };
  const classes = (el) => el.className.split(" ").filter(Boolean);
  node.classList = {
    add: (...cs) => {
      node.className = [...new Set([...classes(node), ...cs])].join(" ");
    },
    remove: (...cs) => {
      node.className = classes(node).filter((c) => !cs.includes(c)).join(" ");
    },
    contains: (c) => classes(node).includes(c),
    toggle: (c, on) => {
      const has = classes(node).includes(c);
      if (on === undefined ? !has : on) node.classList.add(c);
      else node.classList.remove(c);
    },
  };
  // 真实 DOM 的 append/appendChild 都接受字符串；这里包一层，免得为了测试去改产品代码
  const asNode = (child) =>
    typeof child === "object" && child !== null ? child : { textContent: String(child), children: [] };
  node.appendChild = (child) => {
    const wrapped = asNode(child);
    node.children.push(wrapped);
    wrapped.parent = node;
    return wrapped;
  };
  node.append = (...xs) => xs.forEach((x) => node.appendChild(x));
  node.prepend = (child) => {
    const wrapped = asNode(child);
    node.children.unshift(wrapped);
    wrapped.parent = node;
  };
  node.addEventListener = (ev, fn) => {
    (node.handlers[ev] = node.handlers[ev] || []).push(fn);
  };
  node.click = () => (node.handlers.click || []).forEach((fn) => fn());
  return node;
}

const IDS = ["health", "total", "gate", "hint", "q", "send", "reset", "tokenInput", "tokenBtn", "pricing", "thread"];
const byId = Object.fromEntries(IDS.map((id) => [id, makeEl()]));
const document = {
  getElementById: (id) => byId[id] || makeEl(),
  createElement: (tag) => makeEl(tag),
};
const storage = new Map();
const localStorage = {
  getItem: (k) => (storage.has(k) ? storage.get(k) : null),
  setItem: (k, v) => storage.set(k, v),
};
let askResponse = null; // 由“端到端”那条用例设置
const fetch = async (url) => {
  if (String(url).includes("/api/health")) {
    return {
      ok: true,
      json: async () => ({
        auth: false,
        authorized: true,
        ready: true,
        model: "deepseek-flash",
        index: { usable: true, n_files: 44, n_chunks: 2501, built_at: "2026-10-09T13:28:40" },
        pricing: { text: "当前为低谷计价" },
        limits: { max_question_chars: 500 },
      }),
    };
  }
  if (String(url).includes("/api/ask")) {
    return askResponse || { ok: false, status: 500, json: async () => ({}) };
  }
  return { ok: false, status: 404, json: async () => ({}) };
};

// 把整个内联脚本当函数体执行，末尾追加一句 return，好把那几个函数取出来测
const factory = new Function(
  "document",
  "localStorage",
  "fetch",
  src + "\nreturn { stepLine, handleEvent, renderResult, state, el, ask };"
);
const app = factory(document, localStorage, fetch);

const walk = (node, out = []) => {
  out.push(node);
  (node.children || []).forEach((c) => walk(c, out));
  return out;
};
const find = (ctx, pred) => walk(ctx.card).find(pred);
const textOf = (node) => walk(node).map((n) => n.textContent).join("");

// ---- 1. 阶段进度：必须真的进 DOM（T37）----
{
  const card = makeEl("div");
  const steps = new Map();
  const stepList = makeEl("ul");
  card.appendChild(stepList);
  const ctx = { card, steps, stepList, body: makeEl("div") };

  app.handleEvent({ type: "stage", stage: "retrieve", status: "start", label: "检索资料库", round: 1 }, ctx);
  assert.equal(stepList.children.length, 1, "T37：进度行没有插进 DOM");
  assert.equal(stepList.children[0].className, "run", "进行中的那一步应标 run");

  app.handleEvent(
    { type: "stage", stage: "retrieve", status: "done", label: "检索资料库", round: 1, hits: 10, elapsed_s: 0.05 },
    ctx
  );
  assert.equal(stepList.children.length, 1, "同一步的 done 不该新增一行");
  assert.equal(stepList.children[0].className, "done");
  assert.match(stepList.children[0].textContent, /命中 10 条/);
  assert.match(stepList.children[0].textContent, /0\.05s/);

  app.handleEvent({ type: "stage", stage: "plan", status: "start", label: "盘点信息缺口" }, ctx);
  app.handleEvent({ type: "stage", stage: "answer", status: "start", label: "组织答案" }, ctx);
  assert.equal(stepList.children.length, 3, "三个不同阶段应各占一行");

  // T39 的现场：调用处少传参数时，这里会抛 TypeError 而不是静默什么都不画
  assert.doesNotThrow(
    () => app.handleEvent({ type: "stage", stage: "load", status: "start", label: "加载检索模型" }, ctx),
    "T39：stepLine 的调用处必须传 (stepList, steps, event)"
  );
}

// ---- 2. 追问提示与错误分支 ----
{
  const ctx = { card: makeEl("div"), steps: new Map(), stepList: makeEl("ul"), body: makeEl("div") };
  app.handleEvent({ type: "start", session_id: "sess-1", carried_previous: "上一问是什么" }, ctx);
  assert.equal(app.state.sessionId, "sess-1", "start 事件要记住 session_id（追问靠它）");
  assert.match(textOf(ctx.body), /带上了上一问/);

  app.handleEvent({ type: "error", message: "索引不可用" }, ctx);
  assert.match(textOf(ctx.body), /本次失败：索引不可用/);
}

// ---- 3. 结果渲染：四段、成本耗时、引用点开看原文 ----
const view = {
  question: "刀具磨损有哪些检测方法？",
  retrieval_query: "刀具磨损有哪些检测方法？",
  carried_from: null,
  sections: {
    研究现状: [
      {
        point: "有两种主流做法",
        citations: [
          { text: "刀具/A.pdf:第 2 页", kind: "exact", hit_index: 0 },
          { text: "刀具/B.pdf", kind: "partial", hit_index: null, file_hits: [1] },
          { text: "幽灵.pdf:第 9 页", kind: "unresolved", hit_index: null },
        ],
      },
    ],
    方法对比: [{ point: "方法A", pros: "便宜", cons: "精度差", citations: [] }],
    结论: [],
    可复用点: [],
  },
  insufficient: true,
  note: "资料中没有直接答案，以下是相邻证据。",
  degraded: false,
  reason: "四段式回答已产出",
  plan_reason: "候选已足够",
  thinking_recommended: true,
  citations: { exact: 1, partial: 1, invalid: ["幽灵.pdf:第 9 页"], uncited: 0 },
  candidates: [
    {
      source_path: "刀具/A.pdf", locator: "第 2 页", start: 2, end: 2, doc_type: "pdf",
      channels: "dense+sparse", score: 0.03, dense_rank: 1, sparse_rank: 1,
      text: "这是候选片段的原文全文，点开引用要能看到它。",
    },
    {
      source_path: "刀具/B.pdf", locator: "第 5 页", start: 5, end: 5, doc_type: "pdf",
      channels: "dense", score: 0.02, dense_rank: 2, sparse_rank: null,
      text: "B 的原文。",
    },
  ],
  retrieval: [{ round: 1, query: "刀具磨损有哪些检测方法？", hits: 2, elapsed_s: 0.05 }],
  calls: [
    { stage: "plan", elapsed_s: 0.4, cost: { usd: 0.0004, cny: 0.003 }, prompt_tokens: 100, cached_tokens: 0, completion_tokens: 50 },
    { stage: "answer", elapsed_s: 19.6, cost: { usd: 0.0023, cny: 0.016 }, prompt_tokens: 200, cached_tokens: 128, completion_tokens: 400 },
  ],
  tokens: { prompt: 300, cached: 128, completion: 450 },
  cost: { peak: false, usd: 0.0027, cny: 0.019 },
  elapsed: { total_s: 20.0, retrieval_s: 0.05, model_s: 20.0 },
  logged_to: "data/usage_log.jsonl",
};

{
  // 照 ask() 的搭法：card 里装着 stepList 与 body，答案都挂在 body 上
  const card = makeEl("div");
  const body = makeEl("div");
  card.appendChild(makeEl("ul"));
  card.appendChild(body);
  const ctx = { card, steps: new Map(), stepList: card.children[0], body };
  app.handleEvent({ type: "result", session_id: "sess-1", turn: 1, data: view }, ctx);

  const text = textOf(ctx.body);
  assert.match(text, /耗时 20s/, "必须显示耗时");
  assert.match(text, /检索 0\.05s/);
  assert.match(text, /估算成本 \$0\.0027/, "必须显示估算成本");
  assert.match(text, /低谷价/);
  assert.match(text, /精确可回查/);
  assert.match(text, /编造 1/, "编造的引用数要显示出来");
  assert.match(text, /相邻证据/, "缺口题的提示必须出现（首跑后加的）");
  assert.match(text, /候选不足以支撑这一段/, "空段要有说明，不能静默省略");
  assert.match(text, /方法A/);

  // 引用标签：三种 kind 的样式不同
  const chips = walk(ctx.card).filter((n) => n.className.startsWith("cite "));
  assert.equal(chips.length, 3, "三条引用应各有一个可点的标签");
  const [exactChip, partialChip, ghostChip] = chips;
  assert.ok(!exactChip.className.includes("partial") && !exactChip.className.includes("unresolved"));
  assert.ok(partialChip.className.includes("partial"), "只到文件的引用要有区分");
  assert.ok(ghostChip.className.includes("unresolved"), "回查不到的引用要标红");

  // 点开精确引用 → 展开候选原文全文
  exactChip.click();
  const quote = find({ card: exactChip.parent }, (n) => n.className === "quote");
  assert.ok(quote, "点引用应展开一个 quote 块");
  assert.match(textOf(quote), /这是候选片段的原文全文/, "展开的必须是原文，不是提示词里的截断版");
  assert.match(textOf(quote), /第 2 页/);

  // 只到文件 → 列出该文件本次命中的片段，并说明只能回到文件层面
  partialChip.click();
  assert.match(textOf(partialChip.parent), /只给了文件名/);
  assert.match(textOf(partialChip.parent), /B 的原文。/);

  // 回查不到 → 明说回查不到，不给原文
  ghostChip.click();
  assert.match(textOf(ghostChip.parent), /回查不到原文/);

  // 候选清单也要能展开（人工核对用）
  const details = walk(ctx.card).find((n) => n.tagName === "details");
  assert.ok(details, "候选清单应该有个可折叠的块");
  assert.match(textOf(details), /模型看到的候选（2 条/);
}

// ---- 4. 端到端：SSE 分帧 → 事件分发 → 渲染 ----
// 这一步覆盖的是"连接中断：TypeError"那类事故的现场：错误发生在读流的循环里。
// 故意把第 3 帧从**中间**切开，考验客户端跨 chunk 拼帧（这是最容易写错的地方）。
{
  const frame = (obj) => `data: ${JSON.stringify(obj)}\n\n`;
  const events = [
    { type: "start", session_id: "sess-e2e", question: "q", carried_previous: null },
    { type: "stage", stage: "retrieve", status: "start", label: "检索资料库", round: 1 },
    { type: "stage", stage: "retrieve", status: "done", label: "检索资料库", round: 1, hits: 5, elapsed_s: 0.05 },
    { type: "stage", stage: "plan", status: "start", label: "盘点信息缺口" },
    { type: "result", session_id: "sess-e2e", turn: 1, data: view },
  ];
  const text = events.map(frame).join("");
  const cut = text.indexOf(frame(events[2])) + 20; // 切在第 3 帧中间
  assert.ok(cut > 20 && cut < text.length, "切点要落在正文里，否则这条测试是空转");
  const encoder = new TextEncoder();
  let chunkIndex = 0;
  const chunks = [text.slice(0, cut), text.slice(cut)];
  askResponse = {
    ok: true,
    body: {
      getReader: () => ({
        read: async () =>
          chunkIndex < chunks.length
            ? { done: false, value: encoder.encode(chunks[chunkIndex++]) }
            : { done: true, value: undefined },
      }),
    },
  };

  byId["thread"].children.length = 0;
  byId["q"].value = "刀具磨损有哪些检测方法？";
  await app.ask();

  assert.equal(byId["thread"].children.length, 1, "提问后应出现一张答案卡片");
  assert.equal(byId["q"].value, "", "发送后输入框应清空");
  assert.equal(byId["send"].disabled, false, "结束后按钮必须恢复可用，否则界面就卡死了");

  const card = byId["thread"].children[0];
  const rendered = textOf(card);
  assert.match(rendered, /耗时 20s/, "跨 chunk 拼帧失败的话，结果根本渲染不出来");
  assert.match(rendered, /这是候选片段的原文全文/);
  assert.ok(!/连接中断|本次失败/.test(rendered), "不该出现连接中断或失败提示");
  assert.equal(app.state.sessionId, "sess-e2e");
  assert.ok(
    walk(card).some((n) => n.className === "done" && /命中 5 条/.test(n.textContent)),
    "进度条应留下一行 done 记录"
  );
}

console.log("FRONTEND SMOKE OK");
