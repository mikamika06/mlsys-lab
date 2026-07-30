/**
 * The smallest DOM the panel's script actually touches, as a module so more than
 * one test can drive the real page instead of grepping its source.
 */
// ---- the smallest DOM the page actually touches ---------------------------
class El {
  constructor(id, cls) {
    this.id = id;
    this._html = "";
    this.textContent = "";
    this.value = "";
    this.style = { setProperty() {}, top: "", left: "" };
    this.dataset = {};
    this.children = [];
    this._cls = new Set((cls || "").split(" ").filter(Boolean));
    this.classList = {
      add: (c) => this._cls.add(c),
      remove: (c) => this._cls.delete(c),
      toggle: (c, on) => (on ? this._cls.add(c) : this._cls.delete(c)),
      contains: (c) => this._cls.has(c),
    };
    this.listeners = {};
  }
  get className() { return [...this._cls].join(" "); }
  set className(v) { this._cls = new Set(String(v).split(" ").filter(Boolean)); }
  get innerHTML() { return this._html; }
  set innerHTML(v) { this._html = String(v); }
  addEventListener(k, fn) { (this.listeners[k] = this.listeners[k] || []).push(fn); }
  removeEventListener() {}
  setAttribute(k, v) { this[k] = v; }
  removeAttribute(k) { delete this[k]; }
  hasAttribute(k) { return this[k] !== undefined; }
  getAttribute(k) { return this[k]; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
  closest() { return null; }
  focus() {} select() {} blur() {} click() {}
  getBoundingClientRect() { return { left: 0, right: 800, top: 0, bottom: 600, width: 800, height: 600 }; }
  scrollIntoView() {}
}

const ids = {};
const el = (id, cls) => (ids[id] = ids[id] || new El(id, cls));
["app", "map", "home", "work", "prog", "hsub", "crumb", "back", "find", "findn",
 "findWrap", "gradeBtn", "runBtn", "rndBtn", "pstartBtn", "pstartLabel", "runLabel",
 "ta", "hl", "gutter", "aline", "cwrap", "brief", "metrics", "msList", "cons",
 "chcmd", "verdict", "vbadge", "vlabel", "vsub", "instSt", "fname", "ficon",
 "rzL", "rzR"].forEach((i) => el(i));
el("app").className = "app is-home";

global.document = {
  getElementById: (i) => ids[i] || null,
  querySelector: () => null,
  querySelectorAll: () => [],
  addEventListener() {},
  createElement: () => new El(),
  documentElement: { style: { setProperty() {}, getPropertyValue: () => "300px" } },
  body: new El("body"),
  activeElement: null,
  execCommand: () => false,
};
global.window = {
  addEventListener(k, fn) { (global.window._l = global.window._l || {}), ((global.window._l[k] = global.window._l[k] || []).push(fn)); },
  innerWidth: 1200, innerHeight: 800, getComputedStyle: () => ({ display: "block", getPropertyValue: () => "300px" }),
};
global.getComputedStyle = global.window.getComputedStyle;
global.requestAnimationFrame = (fn) => fn();
global.setTimeout = setTimeout;
global.clearTimeout = clearTimeout;
global.KeyboardEvent = class {};
global.MessageEvent = class {};

const sent = [];
global.acquireVsCodeApi = () => ({
  postMessage: (m) => sent.push(m),
  getState: () => ({}),
  setState: () => {},
});


module.exports = { ids, el, sent };
