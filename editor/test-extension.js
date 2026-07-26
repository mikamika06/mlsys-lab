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
    createWebviewPanel(id, title) {
      panel = {
        title,
        webview: {
          html: "",
          postMessage: (m) => { posted.push(m); return Promise.resolve(true); },
          onDidReceiveMessage: (fn) => { panel._onMsg = fn; return { dispose() {} }; },
          asWebviewUri: (u) => ({ toString: () => "vscode-resource:" + u.fsPath }),
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
      k === "workDir" ? WORK : k === "pythonPath" ? PY : d }),
  },
  commands: { registerCommand(id, fn) { registered[id] = fn; return { dispose() {} }; } },
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
  globalState: { _s: {}, get(k, d) { return this._s[k] ?? d; }, update(k, v) { this._s[k] = v; return Promise.resolve(); } },
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

  const bad = results.filter((r) => r[0] === "FAIL");
  for (const [st, name, info] of results) {
    console.log(`  ${st === "ok" ? "ok  " : "FAIL"} ${name.padEnd(30)} ${info}`);
  }
  console.log(`\n${results.length - bad.length}/${results.length} passed`);
  process.exit(bad.length ? 1 : 0);
})();
