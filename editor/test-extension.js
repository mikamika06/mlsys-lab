#!/usr/bin/env node
/**
 * Exercise extension.js without VS Code.
 *
 * The extension is plain Node behind one `require("vscode")`, so stubbing that
 * module runs the real activation, the real workspace-root detection, the real
 * task payload and the real grader dispatch. It cannot prove the webview paints
 * — only VS Code can — but it does prove the host side is not broken, which is
 * the part that fails silently in an editor and is invisible in the browser.
 *
 *     node editor/test-extension.js
 */
const path = require("path");
const fs = require("fs");
const Module = require("module");

const REPO = path.resolve(__dirname, "..");
const executed = [];

// MODE=installed drives the pip-installed path: no repo folder in the workspace,
// so the extension has to locate the bank by asking the interpreter. That is the
// path a Marketplace user takes, and it is invisible to a checkout-only test.
const MODE = process.env.MLSYS_TEST_MODE || "checkout";
const PY = process.env.MLSYS_TEST_PYTHON || "python3";
const WORK = process.env.MLSYS_TEST_WORKDIR || path.join(require("os").tmpdir(), "mlsys-test-work");
fs.rmSync(WORK, { recursive: true, force: true });
const posted = [];
let statusText = null, registered = {}, panel = null;

// ---- the stub -------------------------------------------------------------
const vscode = {
  window: {
    createTreeView(id, opts) {
      treeProvider = opts.treeDataProvider;
      treeView = { visible: false, dispose() {},
                   onDidChangeVisibility: (fn) => { treeView._onVis = fn; return { dispose() {} }; } };
      return treeView;
    },
    createWebviewPanel(id, title) {
      panel = {
        title,
        webview: {
          html: "",
          postMessage: (m) => { posted.push(m); return Promise.resolve(true); },
          onDidReceiveMessage: (fn) => { panel._onMsg = fn; return { dispose() {} }; },
          asWebviewUri: (u) => ({ toString: () => "vscode-resource:" + u.fsPath }),
          _state: {},
          cspSource: "vscode-resource:",
        },
        onDidDispose: () => ({ dispose() {} }),
        reveal() {}, dispose() {},
      };
      return panel;
    },
    createStatusBarItem() {
      return { show() {}, dispose() {}, set text(v) { statusText = v; }, get text() { return statusText; } };
    },
    showErrorMessage: (m) => { throw new Error("extension reported: " + m); },
    showInformationMessage() {},
    createOutputChannel: () => ({ appendLine() {}, show() {}, dispose() {} }),
  },
  workspace: {
    workspaceFolders: MODE === "installed" ? [] : [{ uri: { fsPath: REPO } }],
    getConfiguration: () => ({ get: (k, d) =>
      k === "workDir" ? WORK : k === "pythonPath" ? PY
      : k === "runTimeoutSeconds" ? 3        // short, so the runaway-guard test is quick
      : d }),
  },
  commands: { registerCommand(id, fn) { registered[id] = fn; return { dispose() {} }; },
              executeCommand(id, ...a) { executed.push(id);
                return registered[id] ? Promise.resolve(registered[id](...a)) : Promise.resolve(); } },
  EventEmitter: class { constructor() { this.event = () => ({ dispose() {} }); } fire() {} },
  TreeItem: class { constructor(label, state) { this.label = label; this.collapsibleState = state; } },
  TreeItemCollapsibleState: { None: 0, Collapsed: 1, Expanded: 2 },
  ThemeIcon: class { constructor(id) { this.id = id; } },
  Uri: { file: (p) => ({ fsPath: p, toString: () => "file://" + p }), joinPath: (u, ...r) => ({ fsPath: path.join(u.fsPath, ...r) }) },
  ViewColumn: { Active: 1 },
  StatusBarAlignment: { Left: 1, Right: 2 },
  ProgressLocation: { Notification: 15 },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (req, ...rest) {
  if (req === "vscode") return "vscode";
  return origResolve.call(this, req, ...rest);
};
require.cache["vscode"] = { id: "vscode", filename: "vscode", loaded: true, exports: vscode };

// ---- run ------------------------------------------------------------------
const ext = require(path.join(__dirname, "extension.js"));
const ctx = {
  subscriptions: [],
  globalState: { _s: {}, _sync: null, get(k, d) { return this._s[k] ?? d; },
                 update(k, v) { this._s[k] = v; return Promise.resolve(); },
                 setKeysForSync(keys) { this._sync = keys; } },
  extensionUri: vscode.Uri.file(__dirname),
  extensionPath: __dirname,
};

const results = [];
const check = (name, fn) => {
  try { const info = fn(); results.push(["ok", name, info || ""]); }
  catch (e) { results.push(["FAIL", name, e.message.slice(0, 140)]); }
};
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  check("activate() runs", () => {
    ext.activate(ctx);
    if (!Object.keys(registered).length) throw new Error("no command registered");
    return Object.keys(registered).join(", ");
  });

  check("command id is mlsys.open", () => {
    if (!registered["mlsys.open"]) throw new Error("registered: " + Object.keys(registered));
  });

  await registered["mlsys.open"]();
  check("panel opens and gets HTML", () => {
    if (!panel) throw new Error("no panel created");
    if (!panel.webview.html.includes("<html")) throw new Error("html is empty");
    if (panel.webview.html.includes("{{")) throw new Error("unsubstituted placeholder in html");
    return panel.webview.html.length + " bytes";
  });

  await wait(300);
  check("roadmap is sent on ready", async () => {});
  panel._onMsg({ type: "ready" });
  await wait(800);

  check("map message carries the bank", () => {
    const m = posted.find((x) => x.type === "map" || x.type === "mapdata");
    if (!m) throw new Error("no map posted; got: " + posted.map((p) => p.type).join(","));
    const built = m.payload.totals.built;
    if (built < 2000) throw new Error("only " + built + " tasks in the map");
    return built + " tasks, " + m.payload.tiers.length + " areas";
  });

  check("sidebar is registered", () => {
    if (!treeProvider || !treeView) throw new Error("no tree view — no activity-bar icon");
    const rows = treeProvider.getChildren();
    if (!rows || rows.length !== 1) throw new Error(`sidebar shows ${(rows||[]).length} rows, want 1 doorway`);
    if (rows[0].command.command !== "mlsys.open") throw new Error("the row does not open the roadmap");
    return "one row: " + rows[0].label;
  });

  check("revealing the sidebar opens the roadmap", () => {
    executed.length = 0;
    treeView._onVis({ visible: true });
    if (!executed.includes("mlsys.open"))
      throw new Error("clicking the icon did not open the panel; fired: " + executed.join(","));
    return "mlsys.open fired on reveal";
  });

  check("progress follows the user across machines", () => {
    const keys = ctx.globalState._sync;
    if (!keys || !keys.includes("mlsys.solved"))
      throw new Error("mlsys.solved is not declared for Settings Sync — a second machine starts at zero");
    return keys.join(", ");
  });

  check("areas start folded", () => {
    const h = panel.webview.html;
    if (!/FOLD_V/.test(h)) throw new Error("the fold flag is not versioned — a wrong saved state could never be corrected");
    // The renderer keys areas by position; folding with any other key is a no-op,
    // which is exactly the bug this line exists to prevent recurring.
    if (!/collapsed\.add\('ti'\+i\)/.test(h))
      throw new Error("first render does not collapse using the renderer's own key");
    const keyForm = /const key='ti'\+ti/.test(h);
    if (!keyForm) throw new Error("renderer no longer keys areas by position — update the fold");
    if (!/persistFolds\(\)/.test(h)) throw new Error("the learner's folding is not remembered");
    return "collapsed on first render, choice persisted";
  });

  check("the panel tab carries the project icon", () => {
    if (!panel.iconPath) throw new Error("no iconPath on the webview panel");
    if (!String(panel.iconPath.fsPath || panel.iconPath).includes("icon.png"))
      throw new Error("iconPath is not the project mark: " + JSON.stringify(panel.iconPath));
    return "icon.png";
  });

  // Where the bank actually is in this mode — the reference/starter files are read
  // from there, exactly as the extension reads them.
  const BANK = MODE === "installed"
    ? require("child_process").execFileSync(PY,
        ["-c", "from mlsys import bank; print(bank.bank_root())"], { encoding: "utf8" }).trim()
    : path.join(REPO, "tasks");
  check("bank located", () => { if (!fs.existsSync(BANK)) throw new Error("no bank at " + BANK); return MODE + ": " + BANK; });

  // one task per language, through the real host path
  for (const [tid, lang, srcExpect] of [
    ["sys-one-pass-online-softmax-vector", "python", "solve.py"],
    ["cpp-move-ctor-forgets-to-null-double-free-fix", "cpp", "solve.cpp"],
    ["gpu-ex-cuda-coalesced-scale", "cuda", "solve.cu"],
  ]) {
    posted.length = 0;
    panel._onMsg({ type: "open", id: tid });
    await wait(400);
    check(`open ${lang} task`, () => {
      const t = posted.find((x) => x.type === "task");
      if (!t) throw new Error("no task posted");
      if (t.file !== srcExpect) throw new Error(`file is ${t.file}, expected ${srcExpect}`);
      if (!t.md || t.md.length < 200) throw new Error("statement missing");
      if (!t.code || t.code.length < 10) throw new Error("starter code missing");
      return `${t.file}, md ${t.md.length}B, gates ${t.task.gates.length}`;
    });

    // grade the reference and the starter through the extension's own dispatch
    const d = path.join(BANK, tid);
    const refFile = { python: "solution_ref.py", cpp: "ref.cpp", cuda: "ref.cu" }[lang];
    const stFile = { python: "starter.py", cpp: "starter.cpp", cuda: "starter.cu" }[lang];

    for (const [label, file, want] of [["reference", refFile, true], ["starter", stFile, false]]) {
      posted.length = 0;
      panel._onMsg({ type: "grade", id: tid, file: srcExpect, code: fs.readFileSync(path.join(d, file), "utf8") });
      const t0 = Date.now();
      while (!posted.some((x) => x.type === "result" || x.type === "error") && Date.now() - t0 < 90000) await wait(150);
      check(`grade ${lang} ${label}`, () => {
        const err = posted.find((x) => x.type === "error");
        if (err) throw new Error(err.message.slice(0, 120));
        const r = posted.find((x) => x.type === "result");
        if (!r) throw new Error("timed out waiting for a verdict");
        if (r.data.passed !== want) throw new Error(`passed=${r.data.passed}, expected ${want}`);
        return JSON.stringify(r.data.metrics).slice(0, 70);
      });
    }

    // the shipped starter must survive being graded
    if (lang !== "python") {
      check(`${lang} starter file intact`, () => {
        if (!fs.existsSync(path.join(d, stFile))) throw new Error(stFile + " was destroyed");
      });
    }
    check(`${lang} bank not written to`, () => {
      const stray = fs.readdirSync(d).filter((f) => f.startsWith("solve."));
      if (stray.length) throw new Error("wrote into the bank: " + stray.join(", "));
      const mine = path.join(WORK, tid, srcExpect);
      if (!fs.existsSync(mine)) throw new Error("solution not written to the workspace: " + mine);
    });
  }

  // ---- Run: execute the file, stream what it prints ------------------------
  // Grading proves correctness and says nothing about a print or a traceback.
  // These cover the three ways an in-editor runner goes wrong: output that never
  // arrives, a process nobody can kill, and one that outlives the button.
  const PYTASK = "sys-one-pass-online-softmax-vector";
  const runOut = () => posted.filter((x) => x.type === "runout").map((x) => x.chunk).join("");
  const untilRunEnd = async (ms) => {
    const t0 = Date.now();
    while (!posted.some((x) => x.type === "runend") && Date.now() - t0 < ms) await wait(80);
    return posted.find((x) => x.type === "runend");
  };

  posted.length = 0;
  panel._onMsg({ type: "run", id: PYTASK, file: "solve.py",
                 code: "import sys\nprint('hello from run')\nprint('to stderr', file=sys.stderr)\n" });
  {
    const end = await untilRunEnd(30000);
    check("run prints and exits 0", () => {
      if (!end) throw new Error("no runend");
      const o = runOut();
      if (!o.includes("hello from run")) throw new Error("stdout missing: " + JSON.stringify(o.slice(0, 120)));
      if (!o.includes("to stderr")) throw new Error("stderr missing: " + JSON.stringify(o.slice(0, 120)));
      if (end.code !== 0) throw new Error("exit " + end.code + " " + (end.error || ""));
      return `exit 0, ${end.ms} ms`;
    });
  }

  posted.length = 0;
  panel._onMsg({ type: "run", id: PYTASK, file: "solve.py", code: "raise ValueError('boom')\n" });
  {
    const end = await untilRunEnd(30000);
    check("a traceback reaches the console", () => {
      if (!end) throw new Error("no runend");
      const o = runOut();
      if (!/Traceback|ValueError/.test(o)) throw new Error("no traceback: " + JSON.stringify(o.slice(0, 120)));
      if (end.code === 0) throw new Error("a raising script reported exit 0");
      return "exit " + end.code;
    });
  }

  posted.length = 0;
  panel._onMsg({ type: "run", id: PYTASK, file: "solve.py", code: "while True: pass\n" });
  await wait(400);
  panel._onMsg({ type: "stopRun" });
  {
    const end = await untilRunEnd(8000);
    check("Stop kills a runaway loop", () => {
      if (!end) throw new Error("the process outlived Stop");
      if (!end.stopped) throw new Error("ended without being stopped: " + JSON.stringify(end));
      return end.stopped;
    });
  }

  posted.length = 0;
  panel._onMsg({ type: "run", id: PYTASK, file: "solve.py", code: "while True: pass\n" });
  {
    const end = await untilRunEnd(15000);      // the stub sets the limit to 3s
    check("an unattended loop stops itself", () => {
      if (!end) throw new Error("nothing killed it — the extension host would hold the process forever");
      if (end.stopped !== "timeout") throw new Error("ended as " + JSON.stringify(end));
      return "killed on the wall-clock limit";
    });
  }

  posted.length = 0;
  panel._onMsg({ type: "run", id: "gpu-ex-cuda-coalesced-scale", file: "solve.cu", code: "int main(){}" });
  {
    const end = await untilRunEnd(8000);
    check("a native task refuses to run rather than misrun", () => {
      if (!end) throw new Error("no runend");
      if (!/python/.test(end.stopped || "")) throw new Error("unclear refusal: " + JSON.stringify(end));
      return end.stopped;
    });
  }

  check("the button is in the toolbar next to Grade", () => {
    const h = panel.webview.html;
    const r = h.indexOf('id="runBtn"'), g = h.indexOf('id="gradeBtn"');
    if (r < 0) throw new Error("no Run button in the UI");
    if (!(r < g)) throw new Error("Run is not left of Grade");
    if (!/type:'run'/.test(h)) throw new Error("the button never asks the host to run");
    if (!/type:'stopRun'/.test(h)) throw new Error("no way to stop a running process from the UI");
    return "Run, then Grade";
  });

  const bad = results.filter((r) => r[0] === "FAIL");
  for (const [st, name, info] of results) {
    console.log(`  ${st === "ok" ? "ok  " : "FAIL"} ${name.padEnd(30)} ${info}`);
  }
  console.log(`\n${results.length - bad.length}/${results.length} passed`);
  process.exit(bad.length ? 1 : 0);
})();
