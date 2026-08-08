#!/usr/bin/env node
/**
 * Run the panel's own script against a DOM stub and assert the roadmap paints.
 *
 * Every earlier test grepped the HTML for markers, which proves a string is
 * present and nothing about whether the page works. Splitting renderMap into
 * paintMap/drawMap for search shipped a roadmap that rendered nothing, and not one
 * of seventy-three checks noticed, because they were all looking at source text.
 *
 *     node editor/test-render.js
 */
const fs = require("fs");
const path = require("path");

const { ids, sent } = require("./test-dom.js");
const HTML = fs.readFileSync(path.join(__dirname, "media", "workspace.html"), "utf8");

// ---- run the page's script ------------------------------------------------
const script = HTML.match(/<script nonce="\{\{nonce\}\}">([\s\S]*?)<\/script>/)[1];
const results = [];
const check = (name, fn) => {
  try { results.push(["ok", name, fn() || ""]); }
  catch (e) { results.push(["FAIL", name, (e.message || String(e)).slice(0, 160)]); }
};

let api = null;
check("the panel script runs without throwing", () => {
  const factory = new Function(script + "\n;return {renderMap, paintMap, drawMap, filterMap, openTask, openProject, gradeMilestone, renderFiles, showLeft};");
  api = factory();
  if (typeof api.renderMap !== "function") throw new Error("renderMap is not exported");
  return "loaded";
});

const MAP = {
  totals: { solved: 2, built: 6, planned: 6 },
  tiers: [
    { key: "gpu-cuda", name: "GPU / CUDA", planned: 3, builtCount: 3, roman: "", tracks: [
      { num: "", name: "Shared-memory bank conflicts", planned: 3, tasks: [
        { id: "gpu-pad-tile", title: "Pad the tile", difficulty: 3, solved: true, native: "cuda" },
        { id: "gpu-swizzle-index", title: "Swizzle the index", difficulty: 4, solved: false, native: "cuda" },
        { id: "gpu-broadcast-lane", title: "Broadcast lane", difficulty: 2, solved: false, native: "cuda" }]}]},
    { key: "python-core", name: "Deep Python", planned: 3, builtCount: 3, roman: "", tracks: [
      { num: "", name: "The GIL and what actually scales", planned: 2, tasks: [
        { id: "pyt-modeled-gil-count", title: "Modeled GIL acquire/release count", difficulty: 4, solved: true, native: "" },
        { id: "pyt-atomic-bytecode", title: "Atomic vs non-atomic bytecode", difficulty: 4, solved: false, native: "" }]},
      { num: "", name: "Generational GC & cycle collection", planned: 1, tasks: [
        { id: "pyt-read-gc-get-count", title: "Read gc.get_count()", difficulty: 1, solved: false, native: "" }]}]},
  ],
};

check("the roadmap paints every task", () => {
  api.renderMap(JSON.parse(JSON.stringify(MAP)));
  const html = ids.map.innerHTML;
  if (!html || html.length < 200) throw new Error(`#map holds ${html.length} bytes`);
  const missing = ["gpu-pad-tile", "gpu-swizzle-index", "gpu-broadcast-lane",
                   "pyt-modeled-gil-count", "pyt-atomic-bytecode", "pyt-read-gc-get-count"]
    .filter((id) => !html.includes(id));
  if (missing.length) throw new Error("not rendered: " + missing.join(", "));
  return `${html.length} bytes, 6 tasks`;
});

check("both areas and all three tracks are on the page", () => {
  const html = ids.map.innerHTML;
  for (const name of ["GPU / CUDA", "Deep Python", "Shared-memory bank conflicts",
                      "The GIL and what actually scales", "Generational GC"]) {
    if (!html.includes(name)) throw new Error("missing " + name);
  }
  return "2 areas, 3 tracks";
});

check("the counters read the whole bank", () => {
  if (!/6/.test(ids.prog.innerHTML)) throw new Error("prog: " + ids.prog.innerHTML);
  if (!/2/.test(ids.hsub.innerHTML) && !/6/.test(ids.hsub.innerHTML))
    throw new Error("hsub: " + ids.hsub.innerHTML);
  return ids.prog.innerHTML.replace(/<[^>]+>/g, "");
});

// the filter is debounced, so drive it directly rather than racing a timer
check("filtering keeps only the matches", () => {
  const before = ids.map.innerHTML;
  const filtered = api.filterMap(MAP, "swizzle");
  api.drawMap(filtered, true);
  const html = ids.map.innerHTML;
  if (!html.includes("gpu-swizzle-index")) throw new Error("the match is not shown");
  if (html.includes("pyt-read-gc-get-count")) throw new Error("a non-match survived");
  if (before === html) throw new Error("nothing changed");
  return "1 of 6";
});

check("painting the full map again brings everything back", () => {
  api.drawMap(MAP, false);
  const html = ids.map.innerHTML;
  const missing = ["gpu-pad-tile", "pyt-read-gc-get-count"].filter((i) => !html.includes(i));
  if (missing.length) throw new Error("lost: " + missing.join(", "));
  return "6 tasks";
});

// The host puts Part-2 projects on the roadmap with their execution tier in the
// same field a task uses for its language. An unknown key reached
// LGN[n].toLowerCase() and threw inside the render loop, so the counters painted
// and the map came out empty: one project erased 2054 tasks.
const WITH_PROJECTS = {
  totals: { solved: 2, built: 8, planned: 8 },
  tiers: [
    { key: "projects", name: "Projects · Part 2", planned: 2, builtCount: 2, roman: "", tracks: [
      { num: "", name: "rw2-vllm-serving", planned: 1, tasks: [
        { id: "project:p-continuous-batching-scheduler", title: "Scheduler", difficulty: 4, solved: false, native: "t0" }]},
      { num: "", name: "rw2-pytorch-applied", planned: 1, tasks: [
        { id: "project:p-torch-compile-latency-regression", title: "p99 went up", difficulty: 4, solved: false, native: "t1" }]}]},
    ...MAP.tiers,
  ],
};

check("a project tier does not take the roadmap down with it", () => {
  api.renderMap(JSON.parse(JSON.stringify(WITH_PROJECTS)));
  const html = ids.map.innerHTML;
  if (!html.includes("gpu-pad-tile")) throw new Error("the tasks are gone: " + html.length + " bytes");
  if (!html.includes("p-continuous-batching-scheduler")) throw new Error("the project is not shown");
  return `${html.length} bytes, projects and tasks together`;
});

check("an unrecognised chip label never throws", () => {
  const odd = JSON.parse(JSON.stringify(MAP));
  odd.tiers[0].tracks[0].tasks[0].native = "wat";
  odd.tiers[0].tracks[0].tasks[1].native = undefined;
  api.renderMap(odd);
  const html = ids.map.innerHTML;
  if (!html.includes("gpu-pad-tile") || !html.includes("gpu-swizzle-index"))
    throw new Error("an unknown label removed the task");
  return "unknown and missing labels both survive";
});

// A black panel: project mode replaces the class that makes the work area
// visible, so its own rule has to set display. It did not, and nothing rendered.
check("project mode keeps the work area visible", () => {
  const css = HTML;
  const m = css.match(/\.app\.is-proj \.work\{([^}]*)\}/);
  if (!m) throw new Error("no .app.is-proj .work rule at all");
  if (!/display\s*:\s*(grid|flex)/.test(m[1])) {
    throw new Error("project mode sets no display on .work: " + m[1]);
  }
  // The column widths come from the default .work grid; project mode only has
  // to make it visible, because is-work — the class that does that — is gone.
  if (!/\.app\.is-proj \.right\{[^}]*display/.test(css)) {
    throw new Error("the milestone column has no display of its own");
  }
  return m[1].slice(0, 60);
});

// The project view, driven rather than grepped. The Files switch used to appear
// only once files existed on disk — that is after Start — so opening a project
// showed no switch and the conclusion was that there was none.
const PROJ = { type: "project", md: "# brief", work: "/tmp/w", started: false,
  project: { id: "p-x", title: "P", area: "a", tier: "T0", difficulty: 3,
    edits: ["aot_tools/graph_labeler.py", "tests/test_regression.py"],
    milestones: [{ n: 1, title: "one", gates: [], done: false },
                 { n: 2, title: "two", gates: [], done: false }] } };

check("a project offers the Task / Files switch before Start", () => {
  api.openProject(PROJ);
  if (ids.tabFiles.style.display === "none") throw new Error("the Files tab is hidden");
  const html = ids.leftFiles.innerHTML;
  if (!html.includes("graph_labeler.py")) throw new Error("the declared files are not listed");
  if (!/press Start/.test(html)) throw new Error("no hint that Start copies them");
  return "switch present, 2 files declared";
});

check("the code column carries the directory", () => {
  api.openProject(PROJ);
  const tree = ids.tree.innerHTML;
  if (!tree) throw new Error("the tree beside the editor is empty");
  if (!tree.includes("graph_labeler.py")) throw new Error("the declared files are not in it");
  return "tree rendered before Start";
});

check("the grade button keeps its word after it is pressed", () => {
  api.openProject(PROJ);
  const before = ids.msList.innerHTML.match(/data-msgrade="1">([^<]*)</)[1];
  if (before.trim() !== "grade") throw new Error(`badge starts as "${before}"`);
  api.gradeMilestone(1);
  const after = ids.msList.innerHTML.match(/data-msgrade="1">([^<]*)</)[1];
  if (!/grade/.test(after)) throw new Error(`badge became "${after}"`);
  return `"${before}" -> "${after}"`;
});

const bad = results.filter((r) => r[0] === "FAIL");
for (const [st, name, info] of results) {
  console.log(`  ${st === "ok" ? "ok  " : "FAIL"} ${name.padEnd(46)} ${info}`);
}

console.log(`\n${results.length - bad.length}/${results.length} passed`);
process.exit(bad.length ? 1 : 0);
