#!/bin/bash
# verify_native.sh <id> — a native task is OK only when the reference passes the
# gates AND the shipped starter fails them. Routes on meta.json "native".
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
id="$1"; d="tasks/$id"
[ -f "$d/meta.json" ] || { echo "TASK_FAIL: no meta.json"; exit 1; }
nat=$(python3 -c "import json;print(json.load(open('$d/meta.json')).get('native',''))")

case "$nat" in
  cpp)
    for f in sol.hpp main.cpp ref.cpp starter.cpp task.md; do
      [ -f "$d/$f" ] || { echo "TASK_FAIL: missing $f"; exit 1; }
    done
    PYTHONPATH=src python3 - "$d" <<'PY'
import json, sys
from mlsys.runners.cpp import grade
d = sys.argv[1]
r = grade(d, "ref.cpp")
if not r.get("passed"):
    print("TASK_FAIL: reference does not pass gates", json.dumps(r.get("metrics", {})), r.get("error", "")[:200]); sys.exit(1)
s = grade(d, "starter.cpp")
if s.get("passed"):
    print("TASK_FAIL: starter unexpectedly passes"); sys.exit(1)
print("TASK_OK")
PY
    ;;
  cuda)
    for f in ref.cu starter.cu check.py task.md; do
      [ -f "$d/$f" ] || { echo "TASK_FAIL: missing $f"; exit 1; }
    done
    PYTHONPATH=src python3 - "$d" <<'PY'
import importlib.util, json, sys
d = sys.argv[1]
spec = importlib.util.spec_from_file_location("chk", d + "/check.py")
chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
gates = json.load(open(d + "/meta.json"))["gates"]
def ok(m):
    for g in gates:
        v = m.get(g["metric"])
        if v is None: return False
        if g["op"] == "<=" and not v <= g["threshold"]: return False
        if g["op"] == ">=" and not v >= g["threshold"]: return False
        if g["op"] == "==" and not v == g["threshold"]: return False
    return True
r = chk.grade("ref.cu")
if not ok(r):
    print("TASK_FAIL: reference does not pass gates", json.dumps({k: str(v)[:40] for k, v in r.items()})); sys.exit(1)
s = chk.grade("starter.cu")
if ok(s):
    print("TASK_FAIL: starter unexpectedly passes"); sys.exit(1)
print("TASK_OK")
PY
    ;;
  *) echo "TASK_FAIL: meta.native must be 'cpp' or 'cuda' (got '$nat')"; exit 1 ;;
esac
