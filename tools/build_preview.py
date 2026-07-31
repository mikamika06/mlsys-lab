#!/usr/bin/env python3
"""Standalone BROWSER preview of the bank, grouped by v2 structure:
CATEGORY (area) -> SUBCATEGORY (concept for Part 1, method for Part 2) -> tasks.
Grouping/labels come from src/mlsys/task_list2.json (NOT the model-invented meta.track).
Reuses media/workspace.html; stubs the VS Code API for a plain browser.
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = os.path.join(ROOT, "tasks")
OUTDIR = "/private/tmp/claude-501/-Users-macbook/d5133538-b56b-4545-af01-7c8f7d4d2c96/scratchpad/preview"
os.makedirs(OUTDIR, exist_ok=True)

AREA_TITLE = {
    "python-core": "Deep Python", "cpp-core": "Deep C++", "cpu-perf": "CPU-perf",
    "numeric-tensors": "Numeric & tensors", "algorithms-scratch": "Algorithms (ML)",
    "llm-internals": "LLM internals", "gpu-cuda": "GPU / CUDA", "llm-systems": "LLM systems",
    "rw-applied-quantization": "RW · Quantization", "rw-attention-and-kv": "RW · Attention/KV",
    "rw-compilation-and-export": "RW · Compile/Export", "rw-batching-and-serving": "RW · Batching/Serving",
    "rw-memory-and-offload": "RW · Memory/Offload", "rw-sparsity-pruning-distillation": "RW · Sparsity/Distill",
}
AREA_ORDER = ["python-core", "cpp-core", "cpu-perf", "numeric-tensors", "algorithms-scratch",
              "llm-internals", "gpu-cuda", "llm-systems",
              "rw-applied-quantization", "rw-attention-and-kv", "rw-compilation-and-export",
              "rw-batching-and-serving", "rw-memory-and-offload", "rw-sparsity-pruning-distillation"]
PREFIX_AREA = {"pyt": "python-core", "cpp": "cpp-core", "cpu": "cpu-perf", "num": "numeric-tensors",
               "alg": "algorithms-scratch", "llm": "llm-internals", "gpu": "gpu-cuda", "sys": "llm-systems",
               "rwq": "rw-applied-quantization", "rwa": "rw-attention-and-kv", "rwc": "rw-compilation-and-export",
               "rwb": "rw-batching-and-serving", "rwm": "rw-memory-and-offload", "rws": "rw-sparsity-pruning-distillation"}

# id -> (area, subcategory) from the v2 list
id2 = {}
for r in json.load(open(os.path.join(ROOT, "src/mlsys/task_list2.json")))["rows"]:
    sub = (r.get("method") or r.get("concept") or "other").strip() if r["part"] == 2 else (r.get("concept") or "other").strip()
    id2[r["id"]] = (r["area"], sub)


def area_sub(tid):
    if tid in id2:
        return id2[tid]
    return PREFIX_AREA.get(tid.split("-")[0], "?"), "other"


tasks = {}
groups = {}   # area -> subcat -> [summaries]
for d in sorted(os.listdir(TASKS)):
    mf = os.path.join(TASKS, d, "meta.json")
    if not os.path.isfile(mf):
        continue
    try:
        meta = json.load(open(mf, encoding="utf-8"))
    except Exception:
        continue
    tid = meta.get("id", d)
    area, sub = area_sub(tid)
    md_langs = {}
    try: md_langs["en"] = open(os.path.join(TASKS, d, "task.md"), encoding="utf-8").read()
    except Exception: md_langs["en"] = ""
    for fn in os.listdir(os.path.join(TASKS, d)):
        mm = re.match(r"task\.([a-z]{2})\.md$", fn)
        if mm:
            try: md_langs[mm.group(1)] = open(os.path.join(TASKS, d, fn), encoding="utf-8").read()
            except Exception: pass
    # Native tracks ship real sources, not python: cpp tasks are edited in
    # solve.cpp (against the sol.hpp contract), cuda tasks in solve.cu.
    native = meta.get("native")
    cands = {"cpp": ("solve.cpp", "ref.cpp"), "cuda": ("solve.cu", "ref.cu")}.get(
        native, ("solve.py", "starter.py"))
    code, srcfile = "", cands[0]
    for cand in cands:
        try:
            code = open(os.path.join(TASKS, d, cand), encoding="utf-8").read()
            srcfile = cand
            break
        except Exception:
            pass
    contract = ""
    if native == "cpp":
        try: contract = open(os.path.join(TASKS, d, "sol.hpp"), encoding="utf-8").read()
        except Exception: pass
    tasks[tid] = {
        "code": code, "md_langs": md_langs, "srcfile": srcfile,
        "native": native or "", "contract": contract,
        "task": {"id": tid, "title": meta.get("title", tid), "difficulty": meta.get("difficulty"),
                 "genre": meta.get("genre"), "platform": meta.get("platform"),
                 "track": sub, "gates": meta.get("gates", []),
                 "native": native or ""},   # clean subcategory, NOT meta.track
    }
    groups.setdefault(area, {}).setdefault(sub, []).append(
        {"id": tid, "title": meta.get("title", tid), "difficulty": meta.get("difficulty"),
         "solved": False, "native": native or ""})

# build map: each AREA is a top group (no artificial roman tiers), subcats are the tracks
tiers = []
built_total = 0
for area in sorted(groups, key=lambda a: AREA_ORDER.index(a) if a in AREA_ORDER else 99):
    trs = []
    tcount = 0
    for sub in sorted(groups[area]):
        tk = sorted(groups[area][sub], key=lambda x: (str(x["difficulty"]), str(x["id"])))
        trs.append({"num": "", "name": sub, "planned": len(tk), "tasks": tk})
        tcount += len(tk)
    built_total += tcount
    tiers.append({"roman": "", "key": area, "name": AREA_TITLE.get(area, area),
                  "planned": tcount, "builtCount": tcount, "tracks": trs})

MAP = {"totals": {"solved": 0, "built": built_total, "planned": built_total}, "tiers": tiers}


def esc(s):
    return json.dumps(s, ensure_ascii=False).replace("</", "<\\/")


stub = (
    "<script>\n"
    f"window.__ARENA_DATA={{map:{esc(MAP)},tasks:{esc(tasks)}}};\n"
    "window.__lang='en'; window.__openId=null;\n"
    "function __mdFor(t){return (t.md_langs&&(t.md_langs[window.__lang]||t.md_langs.en))||'';}\n"
    "function __sendTask(id){var t=window.__ARENA_DATA.tasks[id];if(!t)return;\n"
    "  var langs={};for(var k in t.md_langs){langs[k]=t.md_langs[k];}\n"
    "  if(t.contract){for(var k2 in langs){langs[k2]=langs[k2]+'\\n\\n## Contract (sol.hpp)\\n\\n```cpp\\n'+t.contract+'\\n```\\n';}}\n"
    "  window.postMessage({type:'task',file:t.srcfile||'solve.py',code:t.code,md:langs.en||'',md_langs:langs,task:t.task},'*');}\n"
    "window.acquireVsCodeApi=function(){return{postMessage:function(m){if(!m)return;\n"
    "  if(m.type==='ready'){setTimeout(function(){window.postMessage({type:'map',payload:window.__ARENA_DATA.map},'*');},30);}\n"
    "  else if(m.type==='open'&&m.id){window.__openId=m.id;setTimeout(function(){__sendTask(m.id);},20);}\n"
    "  else if(m.type==='grade'){setTimeout(function(){window.postMessage({type:'error',message:'Preview mode — grading runs in VS Code.'},'*');},20);}\n"
    "},getState:function(){return null;},setState:function(){}};};\n"
    "document.addEventListener('DOMContentLoaded',function(){var sel=document.querySelector('.top select')||document.querySelector('select');if(sel){sel.addEventListener('change',function(){window.__lang=sel.value;});}});\n"
    "</script>\n"
)

html = open(os.path.join(ROOT, "extension", "media", "workspace.html"), encoding="utf-8").read()
html = re.sub(r'<meta http-equiv="Content-Security-Policy"[^>]*>', "", html)
html = html.replace("{{nonce}}", "preview").replace("{{katexBase}}", "katex/").replace("{{cspSource}}", "")
html = html.replace("<head>", "<head>\n" + stub, 1)
open(os.path.join(OUTDIR, "index.html"), "w", encoding="utf-8").write(html)

print(f"built preview: {len(tasks)} tasks, {len(tiers)} categories -> {OUTDIR}/index.html")
for t in tiers:
    print(f"  {t['name']}: {t['planned']} tasks in {len(t['tracks'])} subcats")
