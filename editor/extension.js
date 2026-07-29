// mlsys-lab — one webview panel is the whole app. Logs verbosely, because a
// silent extension host failure is otherwise invisible.
const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");
const os = require("os");

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

// ---------------------------------------------------------------------------
// Where the bank is, and where the learner's own files go.
//
// Two ways to have this installed and they must both work: a git checkout (the
// contributor case — grade what you are editing) and `pip install mlsys-lab`,
// which puts the task bank inside site-packages. The installed bank is
// read-only in practice, so the learner's solve.* never goes next to the task;
// it goes in a workspace directory that is theirs to keep.
// ---------------------------------------------------------------------------
function pythonPath() {
  return vscode.workspace.getConfiguration("mlsys").get("pythonPath", "python3");
}

function workDir() {
  const cfg = vscode.workspace.getConfiguration("mlsys").get("workDir", "");
  return cfg ? cfg.replace(/^~(?=$|[/\\])/, os.homedir()) : path.join(os.homedir(), "mlsys-lab");
}

function solvePath(id, srcfile) {
  return path.join(workDir(), id, srcfile);
}

// Ask the installed package where its bank is. One subprocess, short timeout —
// a wrong pythonPath must fail fast and say so, not hang the panel.
function probeInstalled() {
  try {
    const r = cp.execFileSync(pythonPath(),
      ["-c", "from mlsys import bank; print(bank.bank_root()); print(bank.curriculum_file() or ''); print(bank.projects_root() or '')"],
      { encoding: "utf8", timeout: 20000, stdio: ["ignore", "pipe", "pipe"] });
    const [tasksDir, listFile, projDir] = r.trim().split("\n");
    if (!tasksDir || !fs.existsSync(tasksDir)) return null;
    return { mode: "installed", tasksDir,
             listFile: listFile && fs.existsSync(listFile) ? listFile : null,
             projectsDir: projDir && fs.existsSync(projDir) ? projDir : null, pyEnv: {} };
  } catch (e) {
    log("probeInstalled failed: " + (e.stderr || e.message || "").toString().slice(0, 300));
    return null;
  }
}

function labFromCheckout() {
  const root = repoRoot();
  if (!root || !fs.existsSync(path.join(root, "tasks"))) return null;
  const listFile = path.join(root, "src", "mlsys", "task_list2.json");
  const projDir = path.join(root, "projects");
  return {
    mode: "checkout", tasksDir: path.join(root, "tasks"),
    listFile: fs.existsSync(listFile) ? listFile : null,
    projectsDir: fs.existsSync(projDir) ? projDir : null,
    pyEnv: { PYTHONPATH: path.join(root, "src") },   // src-layout: engine is not installed
  };
}

async function ensureLab() {
  const checkout = labFromCheckout();
  if (checkout) { log("bank: checkout at " + checkout.tasksDir); return checkout; }

  const installed = probeInstalled();
  if (installed) { log("bank: installed at " + installed.tasksDir); return installed; }

  const pick = await vscode.window.showInformationMessage(
    "mlsys-lab needs its task bank. Install the mlsys-lab package (about 11 MB) with pip?",
    { modal: false }, "Install", "Choose interpreter", "Cancel");
  if (pick === "Choose interpreter") {
    await vscode.commands.executeCommand("workbench.action.openSettings", "mlsys.pythonPath");
    return null;
  }
  if (pick !== "Install") return null;

  const res = await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: "Installing mlsys-lab…" },
    () => new Promise((resolve) => {
      const p = cp.spawn(pythonPath(), ["-m", "pip", "install", "--upgrade", "mlsys-lab"]);
      let err = "";
      p.stderr.on("data", (d) => { err += d.toString(); log(d.toString().trimEnd()); });
      p.stdout.on("data", (d) => log(d.toString().trimEnd()));
      p.on("error", (e) => { log("pip spawn failed: " + e.message); resolve({ ok: false, err: e.message }); });
      p.on("close", (c) => {
        if (c !== 0) log("pip exited " + c + "\n" + err.slice(-800));
        resolve({ ok: c === 0, err });
      });
    }));
  if (!res.ok) {
    // A Homebrew or distro python refuses to install into itself (PEP 668). That is
    // the most likely failure on macOS and Debian, and "pip install failed" sends
    // the reader nowhere, so name the actual cause and the actual fix.
    if (/externally-managed-environment|externally managed/i.test(res.err || "")) {
      const pick = await vscode.window.showErrorMessage(
        "This Python refuses to install packages into itself (PEP 668). Use a virtual environment, "
        + "then point mlsys.pythonPath at it.", "How", "Open settings");
      if (pick === "Open settings") {
        await vscode.commands.executeCommand("workbench.action.openSettings", "mlsys.pythonPath");
      } else if (pick === "How") {
        log("python3 -m venv ~/.mlsys-venv && ~/.mlsys-venv/bin/pip install mlsys-lab");
        log("then set mlsys.pythonPath to ~/.mlsys-venv/bin/python");
        out && out.show(true);
      }
      return null;
    }
    vscode.window.showErrorMessage(
      `mlsys-lab: pip install failed. See the mlsys-lab output channel. You can also run: ${pythonPath()} -m pip install mlsys-lab`);
    return null;
  }
  const after = probeInstalled();
  if (!after) vscode.window.showErrorMessage("mlsys-lab: installed, but the bank still could not be located.");
  return after;
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
function curriculum(lab) {
  if (_curriculum) return _curriculum;
  _curriculum = {};
  try {
    if (!lab.listFile) throw new Error("no curriculum file");
    const rows = JSON.parse(fs.readFileSync(lab.listFile, "utf8")).rows;
    for (const r of rows) {
      // A track with no name used to render as a group literally called "other",
      // which reads as broken rather than as unclassified. The area's own name is
      // both true and useful, so an unnamed track is absorbed into it.
      const sub = ((r.method || r.concept || "") + "").trim() || AREA_TITLE[r.area] || r.area;
      _curriculum[r.id] = [r.area, sub];
    }
  } catch (_) { /* a checkout without the list still works, via the prefix */ }
  return _curriculum;
}

function placeOf(lab, id) {
  const c = curriculum(lab)[id];
  if (c) return c;
  const area = PREFIX_AREA[id.split("-")[0]] || "llm-systems";
  return [area, AREA_TITLE[area] || area];
}

function scanBuilt(lab) {
  const dir = lab && lab.tasksDir;
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

function homeData(lab, context) {
  const built = scanBuilt(lab);
  const projects = scanProjects(lab);
  const solved = new Set(context.globalState.get("mlsys.solved", []));
  const groups = {};
  for (const b of built) {
    const [area, sub] = placeOf(lab, b.id);
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
  if (projects.length) {
    const tracks = {};
    for (const { spec } of projects) {
      const done = new Set(context.globalState.get("mlsys.milestones." + spec.id, []));
      const n = (spec.milestones || []).length;
      const key = spec.area || "other";
      (tracks[key] = tracks[key] || []).push({
        id: PROJECT_PREFIX + spec.id, title: spec.title, difficulty: spec.difficulty,
        solved: n > 0 && done.size >= n, native: (spec.tier || "T0").toLowerCase(),
      });
    }
    tiers.unshift({
      roman: "", key: "projects", name: "Проєкти · Part 2",
      planned: projects.length, builtCount: projects.length,
      tracks: Object.keys(tracks).sort().map((k) => ({
        num: "", name: k, planned: tracks[k].length, tasks: tracks[k] })),
    });
    total += projects.length;
  }
  return { totals: { solved: built.filter((b) => solved.has(b.id)).length,
                     built: total, planned: total }, tiers };
}


function postWS(msg) { try { panel && panel.webview.postMessage(msg); } catch (_) {} }

// ---------------------------------------------------------------------------
// Part-2 projects. A project is several files and a real repo layout, so the
// editor for it is VS Code itself — the panel carries the ticket, the milestone
// list and the verdicts, and the learner edits the copy in their own workspace.
// ---------------------------------------------------------------------------
const PROJECT_PREFIX = "project:";

function scanProjects(lab) {
  const dir = lab && lab.projectsDir;
  const out = [];
  if (!dir || !fs.existsSync(dir)) return out;
  for (const d of fs.readdirSync(dir).sort()) {
    const pf = path.join(dir, d, "project.json");
    if (!fs.existsSync(pf)) continue;
    try {
      const spec = JSON.parse(fs.readFileSync(pf, "utf8"));
      out.push({ dir: path.join(dir, d), spec });
    } catch (_) { /* a malformed project is skipped, not fatal */ }
  }
  return out;
}

function projectWorkDir(id) { return path.join(workDir(), id); }

// Settings Sync only carries keys that were declared, and a project's key does not
// exist until its first milestone is cleared — declaring once at activation would
// have synced task progress and silently dropped project progress.
function declareSyncKeys(context) {
  try {
    const keys = ["mlsys.solved"];
    for (const k of context.globalState.keys()) {
      if (k.startsWith("mlsys.milestones.")) keys.push(k);
    }
    context.globalState.setKeysForSync(keys);
  } catch (_) { /* an older VS Code without globalState.keys(): task progress still syncs */ }
}

function sendProject(context, lab, id) {
  const found = scanProjects(lab).find((p) => p.spec.id === id);
  if (!found) { postWS({ type: "error", message: "no project " + id }); return; }
  const { dir, spec } = found;
  let md = "";
  try { md = fs.readFileSync(path.join(dir, "brief.md"), "utf8"); } catch (_) {}
  const done = new Set(context.globalState.get("mlsys.milestones." + spec.id, []));
  postWS({ type: "project", md, work: projectWorkDir(spec.id),
    started: fs.existsSync(projectWorkDir(spec.id)),
    project: {
      id: spec.id, title: spec.title, tier: spec.tier, area: spec.area,
      difficulty: spec.difficulty, edits: spec.edits || [],
      milestones: (spec.milestones || []).map((m) => ({
        n: m.n, title: m.title, gates: m.gates || [], done: done.has(m.n) })),
    } });
}

function startProject(context, lab, id) {
  const found = scanProjects(lab).find((p) => p.spec.id === id);
  if (!found) return;
  const py = pythonPath();
  const env = Object.assign({}, process.env, lab.pyEnv || {});
  const args = ["-c",
    "import sys;from mlsys.runners import project as p;print(p.start(sys.argv[1], sys.argv[2]))",
    found.dir, workDir()];
  const r = cp.spawnSync(py, args, { encoding: "utf8", env });
  const dest = (r.stdout || "").trim() || projectWorkDir(id);
  log("project start -> " + dest + (r.stderr ? " :: " + r.stderr.slice(0, 200) : ""));
  try {
    const uri = vscode.Uri.file(path.join(dest, "brief.md"));
    vscode.window.showTextDocument(uri, { preview: false, viewColumn: vscode.ViewColumn.Beside });
  } catch (_) {}
  sendProject(context, lab, id);
}

function gradeProject(context, lab, id, milestone) {
  const found = scanProjects(lab).find((p) => p.spec.id === id);
  if (!found) return;
  const work = projectWorkDir(id);
  if (!fs.existsSync(work)) { postWS({ type: "error", message: "start the project first" }); return; }
  const env = Object.assign({}, process.env, lab.pyEnv || {});
  const args = ["-m", "mlsys.runners.project", found.dir, work];
  if (milestone) args.push(String(milestone));
  const proc = cp.spawn(pythonPath(), args, { cwd: work, env });
  let buf = "", eb = "";
  proc.stdout.on("data", (d) => (buf += d.toString()));
  proc.stderr.on("data", (d) => (eb += d.toString()));
  proc.on("error", (e) => postWS({ type: "error", message: e.message }));
  proc.on("close", () => {
    let data = null;
    try { data = JSON.parse(buf.slice(buf.indexOf("{"))); } catch (_) {}
    if (!data) { postWS({ type: "error", message: (eb + buf).trim().slice(-800) || "no output" }); return; }
    const key = "mlsys.milestones." + id;
    const done = new Set(context.globalState.get(key, []));
    if (milestone && data.passed) done.add(milestone);
    for (const r of data.per_milestone || []) if (r.passed) done.add(r.n);
    context.globalState.update(key, Array.from(done).sort((a, b) => a - b));
    declareSyncKeys(context);
    postWS({ type: "projectResult", data, milestone: milestone || null });
    sendProject(context, lab, id);
  });
}

// Every native track edits its own real source file. One table, used by both
// sendTask and gradeCode, so the editor and the grader can never disagree.
const SRCFILE = { cpp: "solve.cpp",   cuda: "solve.cu" };    // the learner's own file
const STARTER = { cpp: "starter.cpp", cuda: "starter.cu" };  // what ships with the task

function sendTask(context, lab, id) {
  const dir = path.join(lab.tasksDir, id);
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
  // The attempt lives in the learner's own workspace, never beside the task: an
  // installed bank sits in site-packages, and even in a checkout the task dir
  // should stay pristine.
  const starter = STARTER[meta.native] || "starter.py";
  try { code = fs.readFileSync(solvePath(id, srcfile), "utf8"); }
  catch (_) { try { code = fs.readFileSync(path.join(dir, starter), "utf8"); } catch (__) {} }
  postWS({ type: "task", file: srcfile, code, md, task: {
    id: meta.id || id, title: meta.title || id, difficulty: meta.difficulty,
    genre: meta.genre, platform: meta.platform, track: meta.track, gates: meta.gates || [], native: meta.native } });
}

function gradeCode(context, lab, id, file, code) {
  const taskDir = path.join(lab.tasksDir, id);
  let meta = {};
  try { meta = JSON.parse(fs.readFileSync(path.join(taskDir, "meta.json"), "utf8")); } catch (_) {}
  const native = meta.native;                       // "cpp" | "cuda" | undefined
  const srcfile = SRCFILE[native] || (file || "solve.py");
  const solve = solvePath(id, srcfile);
  try {
    fs.mkdirSync(path.dirname(solve), { recursive: true });
    fs.writeFileSync(solve, code, "utf8");
  } catch (e) { postWS({ type: "error", message: e.message }); return; }
  const py = pythonPath();
  // In a checkout the engine is not installed, so src/ goes on the path. When the
  // package is installed there is nothing to add.
  const env = Object.assign({}, process.env, lab.pyEnv || {});
  log(`grade ${id} (${native === "cpp" ? "cpp/clang++" : native === "cuda" ? "cuda/software-gpu" : "py"}) → ${solve}`);
  // Absolute paths on both sides: the task dir may be read-only site-packages and
  // the solution is in the learner's workspace. Both runners accept a candidate
  // outside the task directory.
  const cargs =
    native === "cpp"  ? ["-m", "mlsys.runners.cpp",  taskDir, solve]   // real clang++ compile + run
  : native === "cuda" ? ["-m", "mlsys.runners.cuda", taskDir, solve]   // real .cu on the CUDA-C frontend
  :                     ["-m", "mlsys", "grade", taskDir, "--file", solve, "--json"];
  const proc = cp.spawn(py, cargs, { cwd: workDir(), env });
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
      postWS({ type: "mapdata", payload: homeData(lab, context) });
    }
  });
}

// ---------------------------------------------------------------------------
// Run — execute the file, show what it prints.
//
// Grading answers "is this correct"; it says nothing about a print statement or a
// stack trace halfway down. Running is the other half: the same file, executed as
// a script, output streamed as it arrives. It never touches solved state.
//
// Three things make an in-editor runner safe to press: it streams (a long loop is
// visible immediately), it is killable (the button becomes Stop), and it stops on
// its own (a wall-clock limit and an output cap), so neither an infinite loop nor
// an infinite print can wedge the extension host.
// ---------------------------------------------------------------------------
let runProc = null;
const RUN_OUTPUT_CAP = 200000;      // bytes of combined stdout+stderr kept

function runTimeoutMs() {
  const s = Number(vscode.workspace.getConfiguration("mlsys").get("runTimeoutSeconds", 30));
  return Math.max(1, Number.isFinite(s) ? s : 30) * 1000;
}

// Kill the group, not the process. What we spawn is a runner that itself spawns
// the thing being run — a python script, or a compiled a.out. Killing only the
// parent orphans the child, which keeps burning a core with nobody watching it.
function stopRun(reason) {
  const p = runProc;
  if (!p) return;
  runProc = null;
  clearTimeout(p._mlsysTimer);
  try {
    if (process.platform === "win32") cp.spawn("taskkill", ["/pid", String(p.pid), "/T", "/F"]);
    else process.kill(-p.pid, "SIGKILL");     // negative pid: the whole group
  } catch (_) {
    try { p.kill("SIGKILL"); } catch (__) {}
  }
  log("run killed: " + (reason || "stopped"));
  postWS({ type: "runend", stopped: reason || "stopped" });
}

function runCode(context, lab, id, file, code) {
  if (runProc) { postWS({ type: "runout", chunk: "\n[already running]\n" }); return; }
  const taskDir = path.join(lab.tasksDir, id);
  let meta = {};
  try { meta = JSON.parse(fs.readFileSync(path.join(taskDir, "meta.json"), "utf8")); } catch (_) {}
  const srcfile = SRCFILE[meta.native] || file || "solve.py";
  const solve = solvePath(id, srcfile);
  try {
    fs.mkdirSync(path.dirname(solve), { recursive: true });
    fs.writeFileSync(solve, code, "utf8");
  } catch (e) { postWS({ type: "runend", error: e.message }); return; }

  // One command for all three tracks. The runner knows how each is actually run —
  // a script, clang++ against the task's driver, or the software GPU — and prints
  // the command it used, so nothing here has to know. Unbuffered because python
  // buffers stdout hard when it is a pipe, and a 4 KB buffer means a slow loop
  // shows nothing at all until it exits.
  const env = Object.assign({}, process.env, lab.pyEnv || {}, { PYTHONUNBUFFERED: "1" });

  let proc;
  try {
    proc = cp.spawn(pythonPath(), ["-m", "mlsys.runners.run", taskDir, solve], {
      cwd: path.dirname(solve), env,
      detached: process.platform !== "win32",   // its own group, so Stop can kill all of it
    });
  } catch (e) { postWS({ type: "runend", error: e.message }); return; }
  runProc = proc;
  log(`run ${id} → ${solve}`);

  const t0 = Date.now();
  let sent = 0, capped = false;
  const push = (d) => {
    if (capped) return;
    let s = d.toString();
    // An installed bank older than the runner has no idea what Run is. The raw
    // ImportError names a module the learner has never heard of, so it is
    // translated into the one action that fixes it.
    if (/No module named ['"]?mlsys\.runners\.run/.test(s)) {
      s = `Run needs a newer mlsys-lab package.\n\n    ${pythonPath()} -m pip install --upgrade mlsys-lab\n`;
    }
    if (sent + s.length > RUN_OUTPUT_CAP) { s = s.slice(0, RUN_OUTPUT_CAP - sent); capped = true; }
    sent += s.length;
    if (s) postWS({ type: "runout", chunk: s });
    if (capped) {
      postWS({ type: "runout", chunk: `\n… output capped at ${RUN_OUTPUT_CAP / 1000} KB` });
      stopRun("output limit");
    }
  };
  proc._mlsysTimer = setTimeout(() => {
    postWS({ type: "runout", chunk: `\n… no exit after ${runTimeoutMs() / 1000}s` });
    stopRun("timeout");
  }, runTimeoutMs());

  proc.stdout.on("data", push);
  proc.stderr.on("data", push);
  proc.on("error", (e) => {
    if (runProc !== proc) return;
    runProc = null; clearTimeout(proc._mlsysTimer);
    postWS({ type: "runend", error: e.message });
  });
  proc.on("close", (c) => {
    if (runProc !== proc) return;              // already killed; runend was sent then
    runProc = null; clearTimeout(proc._mlsysTimer);
    postWS({ type: "runend", code: c, ms: Date.now() - t0 });
  });
}

function openPanel(context, lab) {
  if (panel) { try { panel.reveal(vscode.ViewColumn.Active, false); } catch (_) {} log("reveal existing panel"); return; }
  log("creating webview panel…");
  panel = vscode.window.createWebviewPanel(
    "mlsys.workspace", "mlsys-lab", vscode.ViewColumn.Active,
    { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [context.extensionUri] }
  );
  panel.iconPath = vscode.Uri.joinPath(context.extensionUri, "icon.png");
  const nonce = getNonce();
  const p = path.join(context.extensionPath, "media", "workspace.html");
  let doc;
  try {
    const katexBase = panel.webview.asWebviewUri(vscode.Uri.joinPath(context.extensionUri, "media", "katex")).toString() + "/";
    let version = "";
    try { version = JSON.parse(fs.readFileSync(path.join(context.extensionPath, "package.json"), "utf8")).version || ""; } catch (_) {}
    doc = fs.readFileSync(p, "utf8")
      .replace(/{{nonce}}/g, nonce)
      .replace(/{{katexBase}}/g, katexBase)
      .replace(/{{version}}/g, version)
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
    if (m.type === "ready") { log("webview ready → sending map"); postWS({ type: "map", payload: homeData(lab, context) }); }
    else if (m.type === "diag") { log("DIAG: " + m.msg); }
    else if (m.type === "open" && m.id) {
      if (String(m.id).startsWith(PROJECT_PREFIX)) sendProject(context, lab, String(m.id).slice(PROJECT_PREFIX.length));
      else sendTask(context, lab, m.id);
    }
    else if (m.type === "startProject" && m.id) startProject(context, lab, m.id);
    else if (m.type === "gradeProject" && m.id) gradeProject(context, lab, m.id, m.milestone);
    else if (m.type === "openFile" && m.path) {
      try { vscode.window.showTextDocument(vscode.Uri.file(m.path), { preview: false }); } catch (_) {}
    }
    else if (m.type === "grade" && m.id) gradeCode(context, lab, m.id, m.file, m.code || "");
    else if (m.type === "run" && m.id) runCode(context, lab, m.id, m.file, m.code || "");
    else if (m.type === "stopRun") stopRun("stopped");
  });
  panel.onDidDispose(() => { stopRun("panel closed"); panel = null; log("panel disposed"); });
}


// ---------------------------------------------------------------------------
// The activity-bar view. Without one the extension had no icon in the sidebar and
// could only be reached by typing a command, which is not where anyone looks.
//
// The tree resolves the bank lazily. Locating it can mean running a python
// subprocess and, on a machine without the package, prompting to install it —
// neither belongs in the render path of a sidebar that VS Code opens on startup.
// Until it is resolved the view shows one actionable row.
// ---------------------------------------------------------------------------
class BankTree {
  constructor(context) {
    this.context = context;
    this.lab = null;
    this.redirectOnly = true;   // the sidebar is a doorway to the panel, not a menu
    this._onDidChange = new vscode.EventEmitter();
    this.onDidChangeTreeData = this._onDidChange.event;
  }

  refresh(lab) { if (lab !== undefined) this.lab = lab; _curriculum = null; this._onDidChange.fire(); }

  getTreeItem(e) { return e; }

  getChildren(node) {
    // The container exists to put an icon in the activity bar; the roadmap in the
    // panel is the real UI. A second, worse copy of it in a 250px sidebar is not
    // worth maintaining, so the view holds one row and the reveal opens the panel.
    if (this.redirectOnly) {
      const it = new vscode.TreeItem("Open the roadmap", vscode.TreeItemCollapsibleState.None);
      it.command = { command: "mlsys.open", title: "Open the roadmap" };
      it.iconPath = new vscode.ThemeIcon("map");
      return [it];
    }
    if (!this.lab) {
      const lab = labFromCheckout() || probeInstalled();
      if (lab) this.lab = lab;
    }
    if (!this.lab) {
      const it = new vscode.TreeItem("Locate the task bank…", vscode.TreeItemCollapsibleState.None);
      it.command = { command: "mlsys.open", title: "Locate the task bank" };
      it.iconPath = new vscode.ThemeIcon("search");
      it.tooltip = "Opens a checkout, or offers to pip install mlsys-lab";
      return [it];
    }

    const data = homeData(this.lab, this.context);
    if (!node) {
      return data.tiers.map((t) => {
        const solved = t.tracks.reduce((n, tr) => n + tr.tasks.filter((x) => x.solved).length, 0);
        const it = new vscode.TreeItem(t.name, vscode.TreeItemCollapsibleState.Collapsed);
        it.description = `${solved}/${t.planned}`;
        it.id = "area:" + t.key;
        it._area = t;
        return it;
      });
    }
    if (node._area) {
      return node._area.tracks.map((tr) => {
        const solved = tr.tasks.filter((x) => x.solved).length;
        const it = new vscode.TreeItem(tr.name, vscode.TreeItemCollapsibleState.Collapsed);
        it.description = `${solved}/${tr.tasks.length}`;
        it.id = "track:" + node._area.key + ":" + tr.name;
        it._tasks = tr.tasks;
        return it;
      });
    }
    if (node._tasks) {
      return node._tasks.map((t) => {
        const it = new vscode.TreeItem(t.title || t.id, vscode.TreeItemCollapsibleState.None);
        it.id = "task:" + t.id;
        it.description = (t.native || "py") + (t.difficulty ? " · d" + t.difficulty : "");
        it.tooltip = t.id;
        it.iconPath = new vscode.ThemeIcon(t.solved ? "pass-filled" : "circle-large-outline");
        it.command = { command: "mlsys.openTask", title: "Open", arguments: [t.id] };
        return it;
      });
    }
    return [];
  }
}

function activate(context) {
  // The channel is for diagnosing a silent host failure; forcing it open on
  // every window is noise, so it stays available but hidden.
  out = vscode.window.createOutputChannel("mlsys-lab");
  // Solved tasks live in globalState: private to this user, not attached to any repo,
  // and not shared with anyone. Declaring the key for sync means VS Code's own Settings
  // Sync carries it between that person's machines; without it, a second machine starts
  // from zero, which is not what "your progress is saved" should mean.
  declareSyncKeys(context);
  log("=== mlsys-lab activated ===");
  log("extensionPath = " + context.extensionPath);

  const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  status.text = "$(window) mlsys-lab";
  status.tooltip = "Open the mlsys-lab workspace";
  status.command = "mlsys.open";
  status.show();

  // Activation stays cheap and side-effect free: locating the bank can mean
  // running a python subprocess, and doing that in every window that happens to
  // open would be both slow and rude. It happens when the panel is asked for.
  // Opening a panel unasked hijacks the window, so that never happens either.
  const tree = new BankTree(context);
  const view = vscode.window.createTreeView("mlsys.bank", { treeDataProvider: tree });

  // Clicking the activity-bar icon reveals this view; that is the only signal VS Code
  // gives for "the user clicked our icon", so it is what opens the roadmap. Guarded on
  // visibility going true, or hiding and showing the sidebar would fire it repeatedly.
  let wasVisible = view.visible;
  view.onDidChangeVisibility((e) => {
    const nowVisible = e && typeof e.visible === "boolean" ? e.visible : view.visible;
    if (nowVisible && !wasVisible) vscode.commands.executeCommand("mlsys.open");
    wasVisible = nowVisible;
  });

  const open = async () => {
    const lab = await ensureLab();
    if (!lab) { log("no bank — panel not opened"); return null; }
    tree.refresh(lab);
    openPanel(context, lab);
    return lab;
  };

  context.subscriptions.push(out, status, view,
    vscode.commands.registerCommand("mlsys.open", open),
    vscode.commands.registerCommand("mlsys.refresh", () => tree.refresh()),
    vscode.commands.registerCommand("mlsys.openTask", async (id) => {
      const lab = tree.lab || (await open());
      if (!lab) return;
      if (!panel) openPanel(context, lab);
      // The webview may still be loading; it asks for the map on ready, so a task
      // sent too early is dropped. One retry covers the cold-open case.
      const send = () => sendTask(context, lab, id);
      send();
      setTimeout(send, 600);
    }));

  log("ready; sidebar view `mlsys-lab`, or the status bar item");
}
function deactivate() {}
module.exports = { activate, deactivate };
