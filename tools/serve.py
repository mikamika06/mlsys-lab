#!/usr/bin/env python3
"""Staging server: the real workspace, in a browser, with real grading.

The VS Code extension and this server render the SAME `editor/media/workspace.html`
and speak the same message protocol; the only difference is the transport. In the
extension the webview talks to the host through `acquireVsCodeApi()`; here a small
shim maps those same messages onto HTTP calls, and the grading happens in the real
runners — `clang++` actually compiles, the CUDA front end actually executes.

So this is a staging environment, not a preview: if a task grades here, it grades
in the editor.

    python3 tools/serve.py            # http://127.0.0.1:8777
    python3 tools/serve.py --port 9000 --open
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import pathlib
import re
import socketserver
import sys
import threading
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlsys import jsonsafe   # noqa: E402 — the path above has to be set first

TASKS = ROOT / "tasks"
MEDIA = ROOT / "editor" / "media"

SRCFILE  = {"cpp": "solve.cpp",   "cuda": "solve.cu"}     # the learner's own file
STARTER  = {"cpp": "starter.cpp", "cuda": "starter.cu"}   # what ships with the task

# Grouping for the roadmap. Falls back to the id prefix when a task is not in the
# curriculum list, so a freshly generated task still shows up somewhere sensible.
AREA_TITLE = {
    "python-core": "Deep Python", "cpp-core": "Deep C++", "cpu-perf": "CPU-perf",
    "numeric-tensors": "Numeric & tensors", "algorithms-scratch": "Algorithms (ML)",
    "llm-internals": "LLM internals", "gpu-cuda": "GPU / CUDA", "llm-systems": "LLM systems",
    "rw-applied-quantization": "RW · Quantization", "rw-attention-and-kv": "RW · Attention/KV",
    "rw-compilation-and-export": "RW · Compile/Export",
    "rw-batching-and-serving": "RW · Batching/Serving",
    "rw-memory-and-offload": "RW · Memory/Offload",
    "rw-sparsity-pruning-distillation": "RW · Sparsity/Distill",
}
AREA_ORDER = list(AREA_TITLE)
PREFIX_AREA = {
    "pyt": "python-core", "py": "python-core", "cpp": "cpp-core", "cpu": "cpu-perf",
    "num": "numeric-tensors", "alg": "algorithms-scratch", "llm": "llm-internals",
    "gpu": "gpu-cuda", "sys": "llm-systems", "rwq": "rw-applied-quantization",
    "rwa": "rw-attention-and-kv", "rwc": "rw-compilation-and-export",
    "rwb": "rw-batching-and-serving", "rwm": "rw-memory-and-offload",
    "rws": "rw-sparsity-pruning-distillation",
}


def _curriculum():
    """id -> (area, subcategory), from the curriculum list when it is present."""
    out = {}
    f = ROOT / "src" / "mlsys" / "task_list2.json"
    if f.is_file():
        try:
            for r in json.loads(f.read_text())["rows"]:
                # An unnamed track used to render as a group literally called
                # "other", which reads as broken rather than as unclassified.
                sub = (r.get("method") or r.get("concept") or "").strip() \
                    or AREA_TITLE.get(r["area"], r["area"])
                out[r["id"]] = (r["area"], sub)
        except Exception:
            pass
    return out


def _place(tid):
    """Where a task sits. An id the curriculum has never heard of still belongs to
    an area, and showing it under that area's own name beats a bucket called
    "other"."""
    hit = CURRICULUM.get(tid)
    if hit:
        return hit
    area = PREFIX_AREA.get(tid.split("-")[0], "llm-systems")
    return area, AREA_TITLE.get(area, area)


CURRICULUM = _curriculum()
SOLVED_FILE = ROOT / ".preview" / "solved.json"


def load_solved():
    try:
        return set(json.loads(SOLVED_FILE.read_text()))
    except Exception:
        return set()


def save_solved(s):
    SOLVED_FILE.parent.mkdir(exist_ok=True)
    SOLVED_FILE.write_text(json.dumps(sorted(s)))


def scan():
    """Every task in the bank, grouped as the roadmap expects."""
    solved = load_solved()
    groups, meta_by_id = {}, {}
    for d in sorted(TASKS.iterdir()):
        mf = d / "meta.json"
        if not mf.is_file():
            continue
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid = meta.get("id", d.name)
        meta_by_id[tid] = meta
        area, sub = _place(tid)
        groups.setdefault(area, {}).setdefault(sub, []).append({
            "id": tid, "title": meta.get("title", tid), "difficulty": meta.get("difficulty"),
            "solved": tid in solved, "native": meta.get("native") or "",
        })
    tiers, built = [], 0
    for area in sorted(groups, key=lambda a: AREA_ORDER.index(a) if a in AREA_ORDER else 99):
        tracks, n = [], 0
        for sub in sorted(groups[area]):
            ts = sorted(groups[area][sub], key=lambda x: (str(x["difficulty"]), x["id"]))
            tracks.append({"num": "", "name": sub, "planned": len(ts), "tasks": ts})
            n += len(ts)
        built += n
        tiers.append({"roman": "", "key": area, "name": AREA_TITLE.get(area, area),
                      "planned": n, "builtCount": n, "tracks": tracks})
    return {"totals": {"solved": len(solved), "built": built, "planned": built},
            "tiers": tiers}, meta_by_id


def task_payload(tid):
    d = TASKS / tid
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    native = meta.get("native") or ""
    srcfile = SRCFILE.get(native, "solve.py")

    md = (d / "task.md").read_text(encoding="utf-8") if (d / "task.md").is_file() else ""

    # A C++ task is written against a contract header — show it with the statement.
    if native == "cpp" and (d / "sol.hpp").is_file():
        md += "\n\n## Contract (sol.hpp)\n\n```cpp\n" + (d / "sol.hpp").read_text(encoding="utf-8") + "\n```\n"

    # Show the learner's own attempt if there is one, otherwise the shipped
    # starter. The starter file itself is never handed out as the editable file,
    # so grading can never overwrite it.
    code, chain = "", [srcfile, STARTER.get(native, "starter.py")]
    for cand in chain:
        f = d / cand
        if f.is_file():
            code = f.read_text(encoding="utf-8")
            break

    area, sub = _place(tid)
    return {"file": srcfile, "code": code, "md": md,
            "task": {"id": tid, "title": meta.get("title", tid), "difficulty": meta.get("difficulty"),
                     "genre": meta.get("genre"), "platform": meta.get("platform"),
                     "track": sub, "gates": meta.get("gates", []), "native": native}}


def grade(tid, code):
    """Write the attempt next to the task and run the real grader for its language."""
    d = TASKS / tid
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    native = meta.get("native") or ""
    (d / SRCFILE.get(native, "solve.py")).write_text(code, encoding="utf-8")

    if native == "cpp":
        from mlsys.runners import cpp
        return cpp.grade(str(d), "solve.cpp")
    if native == "cuda":
        from mlsys.runners import cuda
        return cuda.grade(str(d), "solve.cu")

    from mlsys import runner
    from mlsys.task import find_task
    r = runner.grade(find_task(tid, TASKS), "solve.py")
    return {"passed": r.passed, "metrics": r.metrics, "gates": r.gates,
            "error": r.error, "verdict": r.verdict}


# The webview expects `acquireVsCodeApi()`. Here it is, backed by fetch().
SHIM = """
<script>
window.acquireVsCodeApi=function(){
  var post=function(m){ window.postMessage(m,'*'); };
  return {
    postMessage:function(m){
      if(!m) return;
      if(m.type==='ready'){
        fetch('/api/map').then(r=>r.json()).then(d=>post({type:'map',payload:d}));
      } else if(m.type==='open'&&m.id){
        fetch('/api/task/'+encodeURIComponent(m.id)).then(r=>r.json()).then(d=>{
          post({type:'task',file:d.file,code:d.code,md:d.md,task:d.task});
        });
      } else if(m.type==='grade'){
        fetch('/api/grade',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({id:m.id,code:m.code})})
          .then(r=>r.json())
          .then(d=>{
            if(d.error && !d.gates) post({type:'error',message:d.error});
            else { post({type:'result',data:d});
                   if(d.passed) fetch('/api/map').then(r=>r.json()).then(x=>post({type:'mapdata',payload:x})); }
          })
          .catch(e=>post({type:'error',message:String(e)}));
      }
    },
    getState:function(){ try{return JSON.parse(localStorage.getItem('mlsys')||'null');}catch(e){return null;} },
    setState:function(s){ try{localStorage.setItem('mlsys',JSON.stringify(s));}catch(e){} }
  };
};
</script>
"""


def page():
    html = (MEDIA / "workspace.html").read_text(encoding="utf-8")
    html = re.sub(r'<meta http-equiv="Content-Security-Policy"[^>]*>', "", html)
    html = (html.replace("{{nonce}}", "staging")
                .replace("{{katexBase}}", "/media/katex/")
                .replace("{{cspSource}}", ""))
    return html.replace("<head>", "<head>\n" + SHIM, 1)


class Handler(http.server.SimpleHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                return self._send(200, page(), "text/html; charset=utf-8")
            if self.path == "/api/map":
                return self._send(200, jsonsafe.dumps(scan()[0]))
            if self.path.startswith("/api/task/"):
                tid = self.path[len("/api/task/"):]
                from urllib.parse import unquote
                tid = unquote(tid)
                if not (TASKS / tid / "meta.json").is_file():
                    return self._send(404, json.dumps({"error": "no such task"}))
                return self._send(200, jsonsafe.dumps(task_payload(tid)))
            if self.path.startswith("/media/"):
                f = MEDIA / self.path[len("/media/"):].split("?")[0]
                # never serve outside the media directory
                if MEDIA.resolve() not in f.resolve().parents or not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                ext = f.suffix.lower()
                ctype = {".css": "text/css", ".js": "application/javascript",
                         ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
                         ".svg": "image/svg+xml"}.get(ext, "application/octet-stream")
                return self._send(200, f.read_bytes(), ctype)
            return self._send(404, b"not found", "text/plain")
        except Exception as e:  # noqa: BLE001 — a staging server must not die on one bad request
            return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def do_POST(self):
        if self.path != "/api/grade":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            tid, code = req.get("id"), req.get("code", "")
            if not tid or not (TASKS / tid / "meta.json").is_file():
                return self._send(404, json.dumps({"error": "no such task"}))
            res = grade(tid, code)
            if res.get("passed"):
                s = load_solved(); s.add(tid); save_solved(s)
            return self._send(200, jsonsafe.dumps(res, default=str))
        except Exception as e:  # noqa: BLE001
            return self._send(200, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def log_message(self, fmt, *a):
        if "/api/" in (a[0] if a else ""):
            sys.stderr.write("  %s\n" % (fmt % a))


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()

    m, _ = scan()
    print(f"mlsys-lab staging  ·  {m['totals']['built']} tasks  ·  {len(m['tiers'])} areas")
    print(f"http://{a.host}:{a.port}    (real grading: python · clang++ · CUDA front end)")
    if a.open:
        threading.Timer(0.6, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    with Server((a.host, a.port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
