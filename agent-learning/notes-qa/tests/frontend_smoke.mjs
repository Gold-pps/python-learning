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

const IDS = [
  "health", "total", "gate", "gateHint", "hint", "q", "send", "reset",
  "tokenInput", "tokenBtn", "pricing", "thread",
];
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
  src + "\nreturn { stepLine, handleEvent, renderResult, state, el, ask, authHeaders };"
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
// 视图**不在这手写了**：它由 `tests/_web_fakes.py` 从真实 `ask()` 产物导出（T43 的教训：
// 手写的夹具写了 `doc_type`、真实视图当时没有这个字段，于是测试全绿而界面错了）。
// 默认读仓库里那份夹具；pytest 会把"刚生成的那份"的路径作为第一个参数传进来，
// 所以即使有人忘了重新生成，冒烟测的也是**真数据的形状**。
const fixturePath = process.argv[2] || new URL("./fixtures/sample_view.json", import.meta.url);
const view = JSON.parse(readFileSync(fixturePath, "utf8"));

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
  // 编造的引用**要逐条列出来**，不能只给一个计数（T44：只给计数等于"知道有问题、不知道哪条"）
  assert.match(text, /已从正文里摘掉/, "编造引用要从正文摘掉并单独说明");
  assert.match(text, /幽灵\.pdf:第 9 页/, "编造的是哪一条必须列出来");
  assert.match(text, /相邻证据/, "缺口题的提示必须出现（首跑后加的）");
  assert.match(text, /候选不足以支撑这一段/, "空段要有说明，不能静默省略");
  assert.match(text, /方法A/);
  // 导出按钮（H4-2）：贴在结果之后，两个都在
  assert.match(text, /导出 MD/);
  assert.match(text, /导出 JSON/);

  // 引用标签：可点的只有**能回查到的**两种（精确 / 只到文件）。
  // 第三种（unresolved）是服务端的兜底分支，正常数据里**一个都不该有**——
  // 这条断言本身就是 T44 的回归：早先夹具手写了一条编造引用，让一个走不到的分支看起来是绿的。
  const chips = walk(ctx.card).filter((n) => n.className.startsWith("cite "));
  assert.equal(chips.length, 2, "真数据里只该有精确 + 只到文件两种可点引用");
  const [exactChip, partialChip] = chips;
  assert.ok(!exactChip.className.includes("partial") && !exactChip.className.includes("unresolved"));
  assert.ok(partialChip.className.includes("partial"), "只到文件的引用要有区分");
  assert.ok(
    !chips.some((n) => n.className.includes("unresolved")),
    "编造引用不该以可点标签的形式出现在正文里"
  );

  // 点开精确引用 → 展开候选原文全文
  exactChip.click();
  const quote = find({ card: exactChip.parent }, (n) => n.className === "quote");
  assert.ok(quote, "点引用应展开一个 quote 块");
  assert.match(textOf(quote), /这是候选片段的原文全文/, "展开的必须是原文，不是提示词里的截断版");
  assert.match(textOf(quote), /第 2 页/);
  // PDF 要写"页 2"，不能写成"行/段 2"（T43：`doc_type` 漏传时就是这样）
  assert.match(textOf(quote), /页 2/);
  assert.ok(!/行\/段/.test(textOf(quote)), "PDF 候选不该被标成行/段");

  // 只到文件 → 列出该文件本次命中的片段，并说明只能回到文件层面
  partialChip.click();
  assert.match(textOf(partialChip.parent), /只给了文件名/);
  assert.match(textOf(partialChip.parent), /B 的原文。/);

  // 候选清单也要能展开（人工核对用）
  const details = walk(ctx.card).find((n) => n.tagName === "details");
  assert.ok(details, "候选清单应该有个可折叠的块");
  assert.match(textOf(details), /模型看到的候选（2 条/);
}

// ---- 3b. 兜底分支（**正常数据走不到**）----
// 服务端会把编造引用从正文里摘掉，所以 `unresolved` 这一支只在异常数据下才触发。
// 留着它是为了"点不开"时不至于抛异常。这里喂一份**手造**数据把兜底渲染也钉住——
// 但它证明的是"兜底不会崩"，**不是**"界面会展示编造引用"（T44 就是这么被误读的）。
{
  const card = makeEl("div");
  const body = makeEl("div");
  card.appendChild(body);
  const ctx = { card, steps: new Map(), stepList: makeEl("ul"), body };
  const fake = JSON.parse(JSON.stringify(view));
  fake.sections["研究现状"][0].citations.push({
    text: "幽灵.pdf:第 9 页",
    kind: "unresolved",
    hit_index: null,
  });

  app.handleEvent({ type: "result", session_id: "s", turn: 1, data: fake }, ctx);

  const ghost = walk(card).find((n) => n.className.includes("unresolved"));
  assert.ok(ghost, "兜底分支：unresolved 应渲染成可点标签（而不是让整页崩掉）");
  ghost.click();
  assert.match(textOf(ghost.parent), /回查不到/, "点了要说明回查不到，不给原文");
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

// ---- 5. 口令：非 ASCII 必须在"提交前"就被拦住（T40）----
// 现场：使用者把中文口令填进去，自检与提问两个请求**全部**报
// "TypeError: ... String contains non ISO-8859-1 code point"——那不是"口令错"，
// 是请求压根构造不出来。所以两道：不存下来、也不放进请求头。
{
  app.state.token = "";
  byId["tokenInput"].value = "自己定一个口令";
  byId["gateHint"].textContent = "";
  byId["tokenBtn"].click();

  assert.equal(app.state.token, "", "非 ASCII 口令不该被存下来");
  assert.match(byId["gateHint"].textContent, /英文标点/, "要说明为什么不行，别让人对着 TypeError 猜");

  app.state.token = "中文口令";
  assert.equal(app.authHeaders()["X-Token"], undefined, "非 ASCII 口令绝不能进请求头");

  app.state.token = "notes-qa-2026";
  assert.equal(app.authHeaders()["X-Token"], "notes-qa-2026", "正常口令要照常带上");

  byId["tokenInput"].value = "notes-qa-2026";
  byId["tokenBtn"].click();
  assert.equal(app.state.token, "notes-qa-2026", "合法口令应被接受");
  assert.equal(storage.get("notesqa_token"), "notes-qa-2026", "并落到本地存储供下次使用");
}

console.log("FRONTEND SMOKE OK");
