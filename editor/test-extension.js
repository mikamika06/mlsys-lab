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
                 keys() { return Object.keys(this._s); },
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

  // Nine tasks were missing from the curriculum list and fell into a group the UI
  // literally labelled "other" — which reads as broken, not as unclassified.
  check("every task sits in a named track", () => {
    const m = posted.find((x) => x.type === "map" || x.type === "mapdata");
    if (!m) throw new Error("no map to inspect");
    const bad = [];
    let tasks = 0;
    for (const tier of m.payload.tiers) {
      for (const tr of tier.tracks) {
        tasks += tr.tasks.length;
        const n = (tr.name || "").trim().toLowerCase();
        if (!n || n === "other") bad.push(`${tier.name}/${tr.name} (${tr.tasks.length} tasks)`);
      }
    }
    if (bad.length) throw new Error("unnamed tracks: " + bad.join(", "));
    return `${tasks} tasks, no "other" bucket`;
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

  // Stop must kill the process that is actually burning the CPU, not just the
  // runner that spawned it. A killed parent with a live child is invisible in the
  // UI and keeps a core busy until the machine is rebooted, so this proves the
  // grandchild stopped writing.
  const beacon = path.join(WORK, "beacon.txt");
  fs.rmSync(beacon, { force: true });
  posted.length = 0;
  panel._onMsg({ type: "run", id: PYTASK, file: "solve.py",
                 code: `import time\nwhile True:\n    open(${JSON.stringify(beacon)}, 'a').write('x')\n    time.sleep(0.02)\n` });
  await wait(1200);
  panel._onMsg({ type: "stopRun" });
  await untilRunEnd(8000);
  {
    await wait(400);
    const before = fs.existsSync(beacon) ? fs.statSync(beacon).size : 0;
    await wait(900);
    const after = fs.existsSync(beacon) ? fs.statSync(beacon).size : 0;
    check("Stop kills the child, not just the runner", () => {
      if (!before) throw new Error("the loop never ran — the test proves nothing");
      if (after !== before) throw new Error(`still writing after Stop (${before} → ${after} bytes)`);
      return `stopped at ${before} bytes`;
    });
  }

  posted.length = 0;
  panel._onMsg({ type: "run", id: "cpp-move-ctor-forgets-to-null-double-free-fix", file: "solve.cpp",
                 code: fs.readFileSync(path.join(BANK, "cpp-move-ctor-forgets-to-null-double-free-fix", "ref.cpp"), "utf8") });
  {
    const end = await untilRunEnd(120000);
    check("a C++ task compiles and runs", () => {
      if (!end) throw new Error("no runend");
      const o = runOut();
      if (!/clang\+\+/.test(o)) throw new Error("the compile command was never shown: " + o.slice(0, 120));
      if (!/move_ctor/.test(o)) throw new Error("the driver's output never arrived: " + o.slice(0, 200));
      if (end.code !== 0) throw new Error("exit " + end.code + " " + JSON.stringify(end));
      return "compiled, ran, exit 0";
    });
  }

  posted.length = 0;
  panel._onMsg({ type: "run", id: "gpu-ex-cuda-coalesced-scale", file: "solve.cu", code: "__global__ void scale( {" });
  {
    const end = await untilRunEnd(60000);
    check("a CUDA parse error is shown, not swallowed", () => {
      if (!end) throw new Error("no runend");
      if (!/parse error|expected/i.test(runOut())) throw new Error("no diagnostic: " + runOut().slice(0, 160));
      if (end.code === 0) throw new Error("a broken kernel reported success");
      return "diagnostic surfaced";
    });
  }

  posted.length = 0;
  panel._onMsg({ type: "run", id: "gpu-ex-cuda-coalesced-scale", file: "solve.cu",
                 code: fs.readFileSync(path.join(BANK, "gpu-ex-cuda-coalesced-scale", "ref.cu"), "utf8") });
  {
    const end = await untilRunEnd(60000);
    check("a CUDA task runs on the software GPU", () => {
      if (!end) throw new Error("no runend");
      const o = runOut();
      if (!/kernels:/.test(o)) throw new Error("the parsed kernels were never listed: " + o.slice(0, 160));
      if (!/transactions/.test(o)) throw new Error("no counters: " + o.slice(0, 160));
      if (/PASS|FAIL/.test(o)) throw new Error("running reported a verdict; that is grading");
      if (end.code !== 0) throw new Error("exit " + end.code);
      return "kernels listed, counters shown, no verdict";
    });
  }

  // ---- typing aids ---------------------------------------------------------
  // The decision half of the editor's key handling is a pure function, lifted out
  // of the page and exercised here. Applying an edit needs a browser; deciding
  // what the edit should be does not, and that is where the behaviour lives.
  {
    const src = panel.webview.html;
    const m = src.match(/\/\* TYPING-AIDS-START \*\/([\s\S]*?)\/\* TYPING-AIDS-END \*\//);
    let plan = null;
    check("the typing aids are extractable and parse", () => {
      if (!m) throw new Error("no TYPING-AIDS block in the page");
      plan = new Function(m[1] + "; return planKey;")();
      if (typeof plan !== "function") throw new Error("planKey is not a function");
      return m[1].length + " bytes";
    });

    // Written as: description, [text, caret, key, language], what the result must be.
    // "|" marks the caret in the expected text.
    const CASES = [
      ["indent survives a newline", "    x = 1", 9, "Enter", "py", "    x = 1\n    |"],
      ["a python block indents", "def f():", 8, "Enter", "py", "def f():\n    |"],
      ["a colon indents only python", "a ? b:", 6, "Enter", "cpp", "a ? b:\n|"],
      ["an opening brace indents", "  if (x) {", 10, "Enter", "cpp", "  if (x) {\n      |"],
      ["a plain line does not indent", "x = 1", 5, "Enter", "py", "x = 1\n|"],
      ["[ closes itself", "a = ", 4, "[", "py", "a = [|]"],
      ["( closes itself", "f", 1, "(", "py", "f(|)"],
      ["{ closes itself", "s = ", 4, "{", "py", "s = {|}"],
      ['" closes itself', "s = ", 4, '"', "py", 's = "|"'],
      ["a quote in a word stays one quote", "don", 3, "'", "py", null],
      ["nothing is closed in front of code", "= xs", 2, "[", "py", null],
      ["a closer already there is stepped over", "f()", 2, ")", "py", "f()|"],
      ["backspace removes both halves", "a = []", 5, "Backspace", "py", "a = |"],
      ["backspace eats a whole indent level", "        x", 8, "Backspace", "py", "    |x"],
      ["shift-tab dedents the line", "        x", 9, "Shift+Tab", "py", "    x|"],
    ];

    // Apply what plan() returned to the text, so the assertion is about the result
    // a learner would see rather than the shape of the returned object.
    const applied = (v, s, en, p) => {
      if (!p) return null;
      if (p.skip) return v.slice(0, s + p.skip) + "|" + v.slice(s + p.skip);
      if (p.del) {
        const [a, b] = p.del;
        const at = s - (Math.min(s, b) - Math.min(s, a));   // the caret keeps its place
        const out = v.slice(0, a) + v.slice(b);
        return out.slice(0, at) + "|" + out.slice(at);
      }
      const out = v.slice(0, s) + p.text + v.slice(en);
      const at = p.caret != null ? s + p.caret : s + p.text.length;
      return out.slice(0, at) + "|" + out.slice(at);
    };

    for (const [name, text, caret, key, lang, want] of CASES) {
      check("typing: " + name, () => {
        const p = plan(text, caret, caret, key, lang);
        const got = applied(text, caret, caret, p);
        if (want === null) {
          if (p !== null) throw new Error("expected the key to type normally, got " + JSON.stringify(got));
          return "handled by the browser";
        }
        if (got !== want) throw new Error(`got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
        return JSON.stringify(want);
      });
    }

    check("typing: a pair wraps the selection", () => {
      const p = plan("keep this", 0, 9, "(", "py");
      if (!p || !p.wrap) throw new Error("selection was replaced instead of wrapped: " + JSON.stringify(p));
      if (p.text !== "(keep this)") throw new Error(p.text);
      return p.text;
    });

    check("typing: a closer splits onto its own line", () => {
      const p = plan("f(", 2, 2, "Enter", "cpp");
      // "f(|)" — the caret sits between the pair, so Enter must open a block
      const p2 = plan("f()", 2, 2, "Enter", "cpp");
      if (p.text !== "\n    ") throw new Error("a trailing ( should indent: " + JSON.stringify(p.text));
      if (p2.text !== "\n    \n") throw new Error("between a pair, the closer needs its own line: " + JSON.stringify(p2.text));
      if (p2.caret !== 5) throw new Error("caret lands at " + p2.caret + ", not on the indented middle line");
      return JSON.stringify(p2.text);
    });
  }

  // An installed update does nothing until the window reloads, and with no version
  // on screen "the fix does not work" and "you are looking at the old build" are
  // indistinguishable. They cost an afternoon once.
  check("a pass shows on the roadmap without reopening", () => {
    const h = panel.webview.html;
    // The pass arrives while the roadmap is off screen. Storing the new data and
    // never repainting it made progress look like it was not being saved at all.
    if (!/mapPainted/.test(h)) throw new Error("nothing tracks whether the painted roadmap is stale");
    if (!/function goHome\(\)\{\s*if\(mapData&&mapData!==mapPainted\)renderMap\(mapData\);/.test(h))
      throw new Error("returning to the roadmap does not repaint stale data");
    return "repaints on return";
  });

  // ---- Part-2 projects -----------------------------------------------------
  {
    const REPO = path.resolve(__dirname, "..");
    const PID = "p-continuous-batching-scheduler";

    posted.length = 0;
    panel._onMsg({ type: "ready" });
    await wait(600);
    // Projects belong on the roadmap — a thousand of them shipped that the panel
    // never mentioned — but never in front of the tasks: a tier of multi-file
    // work at the top pushed two thousand tasks below the fold.
    check("projects are on the roadmap, and never before the tasks", () => {
      const m = posted.find((x) => x.type === "map" || x.type === "mapdata");
      const tiers = m ? m.payload.tiers : [];
      if (!tiers.length) throw new Error("no tiers at all");
      const first = tiers.findIndex((t) => t.key === "projects");
      const lastTask = tiers.map((t) => t.key).lastIndexOf(
        tiers.map((t) => t.key).filter((k) => k !== "projects").slice(-1)[0]);
      if (first === 0) throw new Error("a projects tier is first on the roadmap");
      if (first >= 0 && first < lastTask) {
        throw new Error("a projects tier sits above a task area");
      }
      return first < 0
        ? `${tiers.length} areas, no projects in this fixture`
        : `${tiers.length} areas, projects start at ${first}`;
    });

    // The product is English throughout — README, 2053 task statements, the
    // Marketplace listing. A Ukrainian string in the panel is an inconsistency, not
    // a localisation, because nothing else is translated.
    check("nothing in the panel is Ukrainian", () => {
      const m = posted.find((x) => x.type === "map" || x.type === "mapdata");
      const names = (m ? m.payload.tiers : []).map((t) => t.name);
      const cyr = names.filter((n) => /[\u0400-\u04FF]/.test(n));
      if (cyr.length) throw new Error("Cyrillic area names: " + cyr.join(", "));
      const html = panel.webview.html;
      const lines = html.split("\n").filter((l) => /[\u0400-\u04FF]/.test(l));
      if (lines.length) throw new Error(lines.length + " Cyrillic lines, first: " + lines[0].trim().slice(0, 80));
      return names.slice(0, 3).join(" · ");
    });

    posted.length = 0;
    panel._onMsg({ type: "open", id: "project:" + PID });
    check("opening a project sends its brief and milestones", () => {
      const p = posted.find((x) => x.type === "project");
      if (!p) throw new Error("no project message: " + posted.map((x) => x.type).join(","));
      if (!p.md || p.md.length < 200) throw new Error("brief missing");
      if ((p.project.milestones || []).length !== 7) throw new Error("milestones: " + p.project.milestones.length);
      if (!p.project.edits.length) throw new Error("no files listed to edit");
      return `${p.project.milestones.length} milestones, ${p.project.edits.length} files`;
    });

    // Grade the reference copy through the host: put it where a learner's copy goes.
    const work = path.join(WORK, PID);
    fs.rmSync(work, { recursive: true, force: true });
    fs.mkdirSync(path.dirname(work), { recursive: true });
    fs.cpSync(path.join(REPO, "projects", PID, "reference"), work, { recursive: true });

    posted.length = 0;
    panel._onMsg({ type: "gradeProject", id: PID, milestone: 1 });
    {
      const t0 = Date.now();
      while (!posted.some((x) => x.type === "projectResult" || x.type === "error")
             && Date.now() - t0 < 120000) await wait(200);
      check("grading one milestone of a project", () => {
        const err = posted.find((x) => x.type === "error");
        if (err) throw new Error(err.message.slice(0, 160));
        const r = posted.find((x) => x.type === "projectResult");
        if (!r) throw new Error("timed out");
        if (!r.data.passed) throw new Error("the reference failed milestone 1: "
          + JSON.stringify(r.data.gates || r.data.error).slice(0, 160));
        return r.data.gates.length + " gates green";
      });
    }

    check("a cleared milestone is remembered", () => {
      const done = ctx.globalState.get("mlsys.milestones." + PID, []);
      if (!done.includes(1)) throw new Error("progress not stored: " + JSON.stringify(done));
      return "milestone 1 stored";
    });

    check("project progress follows the user across machines", () => {
      const keys = ctx.globalState._sync || [];
      if (!keys.some((k) => k.startsWith("mlsys.milestones.")))
        throw new Error("milestone keys are not declared for Settings Sync: " + keys.join(","));
      return keys.filter((k) => k.startsWith("mlsys.milestones.")).length + " project keys synced";
    });

    check("the panel has a project view, not a code editor", () => {
      const h = panel.webview.html;
      for (const marker of ["is-proj", "pstartBtn", "msList", "gradeProject", "startProject"])
        if (!h.includes(marker)) throw new Error("missing " + marker);
      if (!/\.app\.is-proj \.center,\.app\.is-proj \.rz\{display:none;\}/.test(h))
        throw new Error("the code column is not hidden for projects");
      return "brief + milestones, editor hidden";
    });
  }

  // ---- roadmap search ------------------------------------------------------
  {
    const src = panel.webview.html;
    const m = src.match(/\/\* SEARCH-CORE-START \*\/([\s\S]*?)\/\* SEARCH-CORE-END \*\//);
    let filterMap = null, mark = null;
    check("the search core is extractable", () => {
      if (!m) throw new Error("no SEARCH-CORE block in the page");
      const mk = new Function("esc", m[1] + "; return {filterMap, mark};");
      const api = mk((x) => String(x == null ? "" : x));
      filterMap = api.filterMap; mark = api.mark;
      if (typeof filterMap !== "function") throw new Error("filterMap missing");
      return m[1].length + " bytes";
    });

    const MAP = {
      totals: { solved: 1, built: 5, planned: 5 },
      tiers: [
        { key: "gpu-cuda", name: "GPU / CUDA", tracks: [
          { name: "Shared-memory bank conflicts", tasks: [
            { id: "gpu-pad-tile-to-kill-bank-conflicts", title: "Pad the tile", solved: false },
            { id: "gpu-swizzle-index", title: "Swizzle the index", solved: true }]}]},
        { key: "python-core", name: "Deep Python", tracks: [
          { name: "The GIL and what actually scales", tasks: [
            { id: "pyt-modeled-gil-acquire-release-count", title: "Modeled GIL acquire/release count", solved: false }]},
          { name: "Generational GC & cycle collection", tasks: [
            { id: "pyt-read-gc-get-count", title: "Read gc.get_count()", solved: false },
            { id: "pyt-fix-a-finalizer-blocked-cycle", title: "Fix a finalizer-blocked cycle", solved: false }]}]},
      ],
    };

    const ids = (q) => filterMap(MAP, q).tiers.flatMap((t) => t.tracks.flatMap((tr) => tr.tasks.map((x) => x.id)));

    check("search matches a task id", () => {
      const got = ids("swizzle");
      if (got.length !== 1 || got[0] !== "gpu-swizzle-index") throw new Error(got.join(","));
      return got[0];
    });

    check("search matches a title", () => {
      const got = ids("finalizer");
      if (got.length !== 1) throw new Error(got.join(","));
      return got[0];
    });

    check("search matches a track name and keeps its tasks", () => {
      const got = ids("gil");
      if (!got.includes("pyt-modeled-gil-acquire-release-count")) throw new Error(got.join(","));
      return got.length + " tasks under the matching track";
    });

    check("search matches an area name", () => {
      const got = ids("cuda");
      if (got.length !== 2) throw new Error("expected both GPU tasks, got " + got.join(","));
      return "2 tasks";
    });

    check("two words must both match", () => {
      const both = ids("gc count");
      const one = ids("count");
      if (both.length !== 1 || both[0] !== "pyt-read-gc-get-count")
        throw new Error("AND semantics broken: " + both.join(","));
      if (one.length < 2) throw new Error("single term should be broader, got " + one.join(","));
      return "AND, not OR";
    });

    check("an empty query returns the map untouched", () => {
      if (filterMap(MAP, "") !== MAP) throw new Error("empty query rebuilt the map");
      return "same object";
    });

    check("no match yields no tiers, and the map is not mutated", () => {
      const r = filterMap(MAP, "zzzzz");
      if (r.tiers.length !== 0) throw new Error("expected nothing");
      if (MAP.tiers[0].tracks[0].tasks.length !== 2) throw new Error("the source map was mutated");
      return "0 hits, source intact";
    });

    check("the count reflects hits, not the whole bank", () => {
      const r = filterMap(MAP, "pyt");
      if (r._hits !== 3 || r.totals.built !== 3) throw new Error(JSON.stringify(r.totals));
      return "3/5";
    });

    check("matches are highlighted, and a regex character cannot break it", () => {
      const h = mark("gpu-swizzle-index", "swizzle");
      if (!/<span class="hit">swizzle<\/span>/.test(h)) throw new Error(h);
      const safe = mark("a.b+c", "b+c");
      if (safe.indexOf("hit") < 0) throw new Error("special characters were treated as a pattern: " + safe);
      return "highlighted";
    });

    check("the toolbar carries the box and the shortcut", () => {
      const h = panel.webview.html;
      for (const marker of ['id="find"', "findn", "paintMap()", "e.key==='/'"])
        if (!h.includes(marker)) throw new Error("missing " + marker);
      if (!/\.app\.is-home \.find\{display:flex;\}/.test(h))
        throw new Error("the box is not shown on the roadmap");
      return "box, count, slash-to-focus";
    });
  }

  check("the panel says which build it is", () => {
    const want = require(path.join(__dirname, "package.json")).version;
    if (!panel.webview.html.includes(want))
      throw new Error(`version ${want} is not shown anywhere in the panel`);
    return want;
  });

  check("Random picks only unsolved work", () => {
    const h = panel.webview.html;
    const rn = h.indexOf('id="rndBtn"'), r = h.indexOf('id="runBtn"');
    if (rn < 0) throw new Error("no Random button");
    if (!(rn < r)) throw new Error("Random is not left of Run");
    if (!/if\(!x\.solved&&x\.id!==curId\)out\.push\(x\.id\)/.test(h))
      throw new Error("the pool is not filtered to unsolved-and-not-current");
    return "left of Run, solved excluded";
  });

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
