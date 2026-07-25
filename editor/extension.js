// mlsys-lab — one webview panel is the whole app. Logs verbosely, because a
// silent extension host failure is otherwise invisible.
const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");

let panel = null;
let out = null;

function log(m) { try { out && out.appendLine(m); } catch (_) {} }

function repoRoot() {
  const folders = vscode.workspace.workspaceFolders || [];
  for (const f of folders) {
    const p = f.uri.fsPath;
    if (fs.existsSync(path.join(p, "src", "mlsys")) && fs.existsSync(path.join(p, "tasks"))) return p;
  }
  return folders.length ? folders[0].uri.fsPath : null;
}

function repoRoot() {
  const folders = vscode.workspace.workspaceFolders || [];
  for (const f of folders) {
    const p = f.uri.fsPath;
    if (fs.existsSync(path.join(p, "src", "mlsys")) && fs.existsSync(path.join(p, "tasks"))) return p;
  }
  return folders.length ? folders[0].uri.fsPath : null;
}

function getNonce() {
  const c = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let s = ""; for (let i = 0; i < 24; i++) s += c[Math.floor(Math.random() * c.length)]; return s;
}

function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
  return String(Math.round(v * 10000) / 10000);
}

// How the bank is grouped on the roadmap. The curriculum list is the single
// source of truth and `tools/serve.py` reads exactly the same file, so the
// editor and the browser can never disagree about which tasks exist — an
// earlier hardcoded copy of an old plan hid 135 tasks from the editor while the
// browser showed all of them.
const AREA_TITLE = {
  "python-core": "Deep Python", "cpp-core": "Deep C++", "cpu-perf": "CPU-perf",
  "numeric-tensors": "Numeric & tensors", "algorithms-scratch": "Algorithms (ML)",
  "llm-internals": "LLM internals", "gpu-cuda": "GPU / CUDA", "llm-systems": "LLM systems",
  "rw-applied-quantization": "RW · Quantization", "rw-attention-and-kv": "RW · Attention/KV",
  "rw-compilation-and-export": "RW · Compile/Export", "rw-batching-and-serving": "RW · Batching/Serving",
  "rw-memory-and-offload": "RW · Memory/Offload", "rw-sparsity-pruning-distillation": "RW · Sparsity/Distill",
};
const AREA_ORDER = Object.keys(AREA_TITLE);
const PREFIX_AREA = {
  pyt: "python-core", py: "python-core", cpp: "cpp-core", cpu: "cpu-perf",
  num: "numeric-tensors", alg: "algorithms-scratch", llm: "llm-internals",
  gpu: "gpu-cuda", sys: "llm-systems", rwq: "rw-applied-quantization",
  rwa: "rw-attention-and-kv", rwc: "rw-compilation-and-export",
  rwb: "rw-batching-and-serving", rwm: "rw-memory-and-offload",
  rws: "rw-sparsity-pruning-distillation",
};

let _curriculum = null;
function curriculum(root) {
  if (_curriculum) return _curriculum;
  _curriculum = {};
  try {
    const rows = JSON.parse(fs.readFileSync(path.join(root, "docs", "task_list2.json"), "utf8")).rows;
    for (const r of rows) {
      const sub = ((r.method || r.concept || "other") + "").trim();
      _curriculum[r.id] = [r.area, sub];
    }
  } catch (_) { /* a checkout without the list still works, via the prefix */ }
  return _curriculum;
}

function placeOf(root, id) {
  const c = curriculum(root)[id];
  if (c) return c;
  return [PREFIX_AREA[id.split("-")[0]] || "other", "other"];
}

function scanBuilt(root) {
  const dir = root && path.join(root, "tasks");
  const arr = [];
  if (!dir || !fs.existsSync(dir)) return arr;
  for (const d of fs.readdirSync(dir).sort()) {
    const mf = path.join(dir, d, "meta.json");
    if (!fs.existsSync(mf)) continue;
    let meta = {};
    try { meta = JSON.parse(fs.readFileSync(mf, "utf8")); } catch (_) { continue; }
    arr.push({ id: meta.id || d, title: meta.title || d, difficulty: meta.difficulty,
               native: meta.native || "" });
  }
  return arr;
}

function homeData(root, context) {
  const built = scanBuilt(root);
  const solved = new Set(context.globalState.get("mlsys.solved", []));
  const groups = {};
  for (const b of built) {
    const [area, sub] = placeOf(root, b.id);
    ((groups[area] = groups[area] || {})[sub] = groups[area][sub] || []).push({
      id: b.id, title: b.title, difficulty: b.difficulty,
      solved: solved.has(b.id), native: b.native,
    });
  }
  const rank = (a) => { const i = AREA_ORDER.indexOf(a); return i < 0 ? 99 : i; };
  const tiers = [];
  let total = 0;
  for (const area of Object.keys(groups).sort((x, y) => rank(x) - rank(y) || x.localeCompare(y))) {
    const tracks = [];
    let n = 0;
    for (const sub of Object.keys(groups[area]).sort()) {
      const ts = groups[area][sub].sort((p, q) =>
        String(p.difficulty).localeCompare(String(q.difficulty)) || p.id.localeCompare(q.id));
      tracks.push({ num: "", name: sub, planned: ts.length, tasks: ts });
      n += ts.length;
    }
    total += n;
    tiers.push({ roman: "", key: area, name: AREA_TITLE[area] || area,
                 planned: n, builtCount: n, tracks });
  }
  return { totals: { solved: built.filter((b) => solved.has(b.id)).length,
                     built: total, planned: total }, tiers };
}


function postWS(msg) { try { panel && panel.webview.postMessage(msg); } catch (_) {} }

// Every native track edits its own real source file. One table, used by both
// sendTask and gradeCode, so the editor and the grader can never disagree.
const SRCFILE = { cpp: "solve.cpp",   cuda: "solve.cu" };    // the learner's own file
const STARTER = { cpp: "starter.cpp", cuda: "starter.cu" };  // what ships with the task

function sendTask(context, root, id) {
  const dir = path.join(root, "tasks", id);
  let meta = {}, md = "", code = "";
  try { meta = JSON.parse(fs.readFileSync(path.join(dir, "meta.json"), "utf8")); } catch (_) {}
  try { md = fs.readFileSync(path.join(dir, "task.md"), "utf8"); } catch (_) {}
  const srcfile = SRCFILE[meta.native] || "solve.py";   // cpp -> solve.cpp, cuda -> solve.cu
  // A cpp task is written against a contract header; show it, or the learner is
  // implementing signatures they cannot see.
  if (meta.native === "cpp") {
    try { md += "\n\n## Contract (sol.hpp)\n\n```cpp\n" + fs.readFileSync(path.join(dir, "sol.hpp"), "utf8") + "\n```\n"; } catch (_) {}
  }
  // Show the learner's attempt if there is one, else seed it from the shipped
  // starter. The starter file is never edited, so grading cannot destroy it.
  const starter = STARTER[meta.native] || "starter.py";
  try { code = fs.readFileSync(path.join(dir, srcfile), "utf8"); }
  catch (_) { try { code = fs.readFileSync(path.join(dir, starter), "utf8"); } catch (__) {} }
  postWS({ type: "task", file: srcfile, code, md, task: {
    id: meta.id || id, title: meta.title || id, difficulty: meta.difficulty,
    genre: meta.genre, platform: meta.platform, track: meta.track, gates: meta.gates || [], native: meta.native } });
}

function gradeCode(context, root, id, file, code) {
  let meta = {};
  try { meta = JSON.parse(fs.readFileSync(path.join(root, "tasks", id, "meta.json"), "utf8")); } catch (_) {}
  const native = meta.native;                       // "cpp" | "cuda" | undefined
  const srcfile = SRCFILE[native] || (file || "solve.py");
  const solve = path.join(root, "tasks", id, srcfile);
  try { fs.writeFileSync(solve, code, "utf8"); } catch (e) { postWS({ type: "error", message: e.message }); return; }
  const py = vscode.workspace.getConfiguration("mlsys").get("pythonPath", "python3");
  // the engine lives in src/ (src-layout), so the grader subprocess needs it on the path
  const env = Object.assign({}, process.env, { PYTHONPATH: path.join(root, "src") });
  log(`grade ${id} (${native === "cpp" ? "cpp/clang++" : native === "cuda" ? "cuda/software-gpu" : "py"})`);
  const cargs =
    native === "cpp"  ? ["-m", "mlsys.runners.cpp",  path.join("tasks", id), "solve.cpp"]  // real clang++ compile + run
  : native === "cuda" ? ["-m", "mlsys.runners.cuda", path.join("tasks", id), "solve.cu"]   // real .cu on the CUDA-C frontend
  :                     ["-m", "mlsys", "grade", id, "--file", "solve.py", "--json"];
  const proc = cp.spawn(py, cargs, { cwd: root, env });
  let buf = "", eb = "";
  proc.stdout.on("data", (d) => (buf += d.toString()));
  proc.stderr.on("data", (d) => (eb += d.toString()));
  proc.on("error", (e) => postWS({ type: "error", message: e.message }));
  proc.on("close", () => {
    let data = null; try { data = JSON.parse(buf); } catch (_) {}
    if (!data) { postWS({ type: "error", message: (eb + buf).trim() || "no output" }); return; }
    postWS({ type: "result", data });
    if (data.passed) {
      const s = new Set(context.globalState.get("mlsys.solved", []));
      s.add(id); context.globalState.update("mlsys.solved", Array.from(s));
      postWS({ type: "mapdata", payload: homeData(root, context) });
    }
  });
}

function openPanel(context, root) {
  if (panel) { try { panel.reveal(vscode.ViewColumn.Active, false); } catch (_) {} log("reveal existing panel"); return; }
  log("creating webview panel…");
  panel = vscode.window.createWebviewPanel(
    "mlsys.workspace", "mlsys-lab", vscode.ViewColumn.Active,
    { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [context.extensionUri] }
  );
  const nonce = getNonce();
  const p = path.join(context.extensionPath, "media", "workspace.html");
  let doc;
  try {
    const katexBase = panel.webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, "media", "katex")).toString() + "/";
    doc = fs.readFileSync(p, "utf8")
      .replace(/{{nonce}}/g, nonce)
      .replace(/{{katexBase}}/g, katexBase)
      .replace(/{{cspSource}}/g, panel.webview.cspSource);
    log("loaded workspace.html (" + doc.length + " bytes) from " + p);
  } catch (e) {
    log("READ FAILED: " + p + " :: " + e.message);
    doc = `<!DOCTYPE html><html><body style="background:#101216;color:#F2694F;font-family:monospace;padding:24px;font-size:13px">`
      + `<b>mlsys-lab — cannot read UI file</b><br><br>${p}<br>${String(e && e.message)}</body></html>`;
  }
  panel.webview.html = doc;
  panel.webview.onDidReceiveMessage((m) => {
    if (!m) return;
    if (m.type === "ready") { log("webview ready → sending map"); postWS({ type: "map", payload: homeData(root, context) }); }
    else if (m.type === "diag") { log("DIAG: " + m.msg); }
    else if (m.type === "open" && m.id) sendTask(context, root, m.id);
    else if (m.type === "grade" && m.id) gradeCode(context, root, m.id, m.file, m.code || "");
  });
  panel.onDidDispose(() => { panel = null; log("panel disposed"); });
}

function activate(context) {
  out = vscode.window.createOutputChannel("mlsys-lab");
  out.show(true);
  const root = repoRoot();
  log("=== mlsys-lab activated ===");
  log("extensionPath = " + context.extensionPath);
  log("repoRoot = " + root);

  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  status.text = "$(window) mlsys-lab";
  status.tooltip = "Open the mlsys-lab workspace";
  status.command = "mlsys.open";
  status.show();

  context.subscriptions.push(out, status,
    vscode.commands.registerCommand("mlsys.open", () => openPanel(context, root)));

  if (!root) { vscode.window.showErrorMessage("mlsys-lab: open the repository folder (it must contain src/mlsys/ and tasks/)."); return; }
  openPanel(context, root);
}
function deactivate() {}
module.exports = { activate, deactivate };
