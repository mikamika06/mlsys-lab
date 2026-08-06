import json
import os

FIX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "_fixtures"))
TEMPLATES = os.path.join(FIX, "templates")
OLLAMA = os.path.join(FIX, "ollama")


class Tools(list):
    def __str__(self):
        return json.dumps(list(self), separators=(",", ":"), ensure_ascii=False)


class Args(dict):
    def __str__(self):
        return json.dumps(dict(self), separators=(",", ":"), ensure_ascii=False)


def _wrap(ctx):
    c = json.loads(json.dumps(ctx))
    if "tools" in c:
        c["tools"] = Tools(c["tools"])
    for m in c.get("messages", []):
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if "arguments" in fn:
                fn["arguments"] = Args(fn["arguments"])
    return c


def inputs():
    with open(os.path.join(TEMPLATES, "inputs.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return {k: _wrap(v) for k, v in raw.items()}


def _ts(p):
    if p.get("enum"):
        return " | ".join(json.dumps(e) for e in p["enum"])
    t = p.get("type")
    if isinstance(t, list):
        return " | ".join(_ts({"type": x}) for x in t)
    simple = {"string": "string", "number": "number", "integer": "number",
              "boolean": "boolean", "object": "object", "null": "null"}
    if t in simple:
        return simple[t]
    if t == "array":
        return _ts(p["items"]) + "[]" if p.get("items") else "any[]"
    return "any"


def _json(v):
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def funcs():
    return {
        "currentDate": lambda: "2026-08-06",
        "toTypeScriptType": _ts,
        "json": _json,
        "toJson": _json,
        "slice": lambda v, *i: (v if not i else
                                (v[i[0]:] if len(i) == 1 else v[i[0]:i[1]])),
    }


def cases(source):
    name = ("semantics_renderings.json" if source == "semantics"
            else "renderings.json")
    with open(os.path.join(TEMPLATES, name), encoding="utf-8") as f:
        return json.load(f)["cases"]


def template_dir(source):
    return os.path.join(TEMPLATES, "semantics") if source == "semantics" else OLLAMA


def source_of(source, name):
    with open(os.path.join(template_dir(source), name), encoding="utf-8") as f:
        return f.read()


def score(render, picked):
    """Fraction of the selected recorded renderings reproduced byte for byte."""
    data = inputs()
    fns = funcs()
    ok = 0
    total = 0
    for source, case in picked:
        total += 1
        try:
            got = render(source_of(source, case["template"]),
                         data[case["input"]], fns)
        except Exception:  # noqa: BLE001
            continue
        if got == case["expected"]:
            ok += 1
    return (ok / total if total else 0.0), ok, total


def pick(source, templates=None, with_tools=None):
    out = []
    data = inputs()
    for c in cases(source):
        if c.get("error"):
            continue
        if templates and not any(t in c["template"] for t in templates):
            continue
        if with_tools is not None:
            has = bool(data[c["input"]].get("tools"))
            if has != with_tools:
                continue
        out.append((source, c))
    return out
