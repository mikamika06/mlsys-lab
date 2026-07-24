#!/bin/bash
# verify_task.sh <task_id>
# Acceptance criterion for a generated Arena task.
# Prints "TASK_OK" only if: reference solution PASSES, starter FAILS,
# and the reference grade is deterministic across two runs. Otherwise
# prints "TASK_FAIL: <reason>". Leaves solve.py == starter.py at the end.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "TASK_FAIL: no repo"; exit 1; }
id="${1:?usage: verify_task.sh <task_id>}"
d="tasks/$id"
for f in meta.json task.md check.py solution_ref.py starter.py; do
  [ -f "$d/$f" ] || { echo "TASK_FAIL: missing $f"; exit 1; }
done

grade() { PYTHONPATH=src perl -e 'alarm 30; exec @ARGV or die' python3 -m mlsys grade "$id" --file solve.py --json 2>/dev/null; }
passed() { python3 -c "import sys,json;print('1' if json.load(sys.stdin).get('passed') else '0')" 2>/dev/null; }
metrics() { python3 -c "import sys,json;print(json.dumps(json.load(sys.stdin).get('metrics'),sort_keys=True))" 2>/dev/null; }

cp "$d/solution_ref.py" "$d/solve.py"
r1=$(grade); rp=$(printf '%s' "$r1" | passed); m1=$(printf '%s' "$r1" | metrics)
r2=$(grade); m2=$(printf '%s' "$r2" | metrics)
cp "$d/starter.py" "$d/solve.py"
s=$(grade); sp=$(printf '%s' "$s" | passed)
cp "$d/starter.py" "$d/solve.py"   # leave starter in place

[ "$rp" = "1" ] || { echo "TASK_FAIL: reference does not pass gates"; exit 1; }
[ "$sp" = "0" ] || { echo "TASK_FAIL: starter unexpectedly passes (gate does not discriminate)"; exit 1; }
[ -n "$m1" ] && [ "$m1" = "$m2" ] || { echo "TASK_FAIL: reference grade is non-deterministic"; exit 1; }
echo "TASK_OK"
