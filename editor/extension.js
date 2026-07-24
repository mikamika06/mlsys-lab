// Arena — minimal, bulletproof. One webview panel = the whole app. Heavy logging.
const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");

let panel = null;
let out = null;

const CURRICULUM = [
  { roman: "I", name: "Мови та основи", tracks: [
    { num: "01", name: "Deep Python", planned: 34 }, { num: "02", name: "Deep C++", planned: 32 },
    { num: "03", name: "Rust", planned: 28 }, { num: "04", name: "Assembly", planned: 22 } ] },
  { roman: "II", name: "Математичне ядро", tracks: [
    { num: "05", name: "Numerical methods", planned: 26 }, { num: "06", name: "Autograd", planned: 24 } ] },
  { roman: "III", name: "Всередині моделі", tracks: [
    { num: "07", name: "Quantization", planned: 29 }, { num: "08", name: "Transformer internals", planned: 33 },
    { num: "09", name: "Architecture zoo", planned: 28 }, { num: "10", name: "KV-cache", planned: 22 } ] },
  { roman: "IV", name: "Заліза та кернели", tracks: [
    { num: "11", name: "GPU kernels", planned: 40 }, { num: "12", name: "ML compilers", planned: 30 },
    { num: "13", name: "CPU-perf", planned: 28 }, { num: "14", name: "Memory", planned: 26 } ] },
  { roman: "V", name: "Масштаб і подача", tracks: [
    { num: "15", name: "Distributed", planned: 34 }, { num: "16", name: "Data pipeline", planned: 26 },
    { num: "17", name: "Serving", planned: 30 } ] },
  { roman: "VI", name: "Майстерність", tracks: [
    { num: "18", name: "Profiling", planned: 24 }, { num: "19", name: "Platform / grader", planned: 22 },
    { num: "20", name: "Freshness", planned: 19 } ] },
];

function log(m) { try { out && out.appendLine(m); } catch (_) {} }

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

function scanBuilt(root) {
  const dir = root && path.join(root, "tasks");
  const arr = [];
  if (!dir || !fs.existsSync(dir)) return arr;
  for (const d of fs.readdirSync(dir).sort()) {
    const mf = path.join(dir, d, "meta.json");
    if (!fs.existsSync(mf)) continue;
    let meta = {};
    try { meta = JSON.parse(fs.readFileSync(mf, "utf8")); } catch (_) {}
    const tn = (String(meta.track || "").match(/\d+/) || [])[0] || null;
    arr.push({ id: meta.id || d, title: meta.title || d, difficulty: meta.difficulty,
               trackNum: tn, native: meta.native || "" });
  }
  return arr;
}
function homeData(root, context) {
  const built = scanBuilt(root);
  const solved = new Set(context.globalState.get("mlsys.solved", []));
  let bt = 0, pt = 0;
  const tiers = CURRICULUM.map((T) => {
    let bc = 0, pl = 0;
    const tracks = T.tracks.map((tr) => {
      pl += tr.planned;
      const tasks = built.filter((b) => b.trackNum === tr.num)
        .map((b) => ({ id: b.id, title: b.title, difficulty: b.difficulty, solved: solved.has(b.id), native: b.native || "" }));
      bc += tasks.length;
      return { num: tr.num, name: tr.name, planned: tr.planned, tasks };
    });
    bt += bc; pt += pl;
    return { roman: T.roman, key: T.key || String(T.name || "").toLowerCase().replace(/[^a-z0-9]+/g, "-"),
             name: T.name, planned: pl, builtCount: bc, tracks };
  });
  return { totals: { solved: built.filter((b) => solved.has(b.id)).length, built: bt, planned: pt }, tiers };
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
  // The statement may ship translations as task.<lang>.md next to task.md. Send
  // every one we have and let the webview pick — it owns the language selector.
  const mdLangs = {};
  try { mdLangs.en = md = fs.readFileSync(path.join(dir, "task.md"), "utf8"); } catch (_) {}
  try {
    for (const f of fs.readdirSync(dir)) {
      const m = /^task\.([a-z]{2})\.md$/.exec(f);
      if (m) mdLangs[m[1]] = fs.readFileSync(path.join(dir, f), "utf8");
    }
  } catch (_) {}
  const srcfile = SRCFILE[meta.native] || "solve.py";   // cpp -> solve.cpp, cuda -> solve.cu
  // A cpp task is written against a contract header; show it, or the learner is
  // implementing signatures they cannot see.
  if (meta.native === "cpp") {
    try {
      const hpp = "\n\n## Contract (sol.hpp)\n\n```cpp\n" + fs.readFileSync(path.join(dir, "sol.hpp"), "utf8") + "\n```\n";
      md += hpp;
      for (const k of Object.keys(mdLangs)) mdLangs[k] += hpp;
    } catch (_) {}
  }
  // Show the learner's attempt if there is one, else seed it from the shipped
  // starter. The starter file is never edited, so grading cannot destroy it.
  const starter = STARTER[meta.native] || "starter.py";
  try { code = fs.readFileSync(path.join(dir, srcfile), "utf8"); }
  catch (_) { try { code = fs.readFileSync(path.join(dir, starter), "utf8"); } catch (__) {} }
  postWS({ type: "task", file: srcfile, code, md, md_langs: mdLangs, task: {
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
      + `<b>Arena — cannot read UI file</b><br><br>${p}<br>${String(e && e.message)}</body></html>`;
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
  out = vscode.window.createOutputChannel("Arena");
  out.show(true);
  const root = repoRoot();
  log("=== Arena activated ===");
  log("extensionPath = " + context.extensionPath);
  log("repoRoot = " + root);

  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  status.text = "$(window) Arena";
  status.tooltip = "Open Arena workspace";
  status.command = "mlsys.open";
  status.show();

  context.subscriptions.push(out, status,
    vscode.commands.registerCommand("mlsys.open", () => openPanel(context, root)));

  if (!root) { vscode.window.showErrorMessage("mlsys-lab: open the repository folder (it must contain src/mlsys/ and tasks/)."); return; }
  openPanel(context, root);
}
function deactivate() {}
module.exports = { activate, deactivate };
