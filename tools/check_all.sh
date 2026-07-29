#!/bin/bash
# Every check this project has, in one command, ordered cheapest first.
#
# Exists because the checks accumulated one at a time as each caught something that had
# already shipped, and remembering all seven before a commit is not a plan.
#
#   tools/check_all.sh          # everything except the full bank sweep (~2 min)
#   tools/check_all.sh --full   # plus verify_all.py over every task (~2 min more)
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

fail=0
run() {
  printf '\n\033[1m%s\033[0m\n' "$1"; shift
  if "$@"; then :; else fail=1; printf '  \033[31mFAILED\033[0m\n'; fi
}

run "concept pages: structure, word count, links resolve"   python3 tools/check_page.py
run "concept pages: every snippet reproduces its table"     python3 tools/verify_pages.py
run "task statements: examples agree with their reference"  python3 tools/check_examples.py
run "statements: KaTeX parses every math span"              node tools/check_latex.js
run "extension host: both bank modes"                       node editor/test-extension.js
run "projects: reference clears every milestone, skeleton none" python3 tools/verify_project.py

printf '\n\033[1mno scratch files left in the bank\033[0m\n'
n=$(find tasks -maxdepth 2 -name 'solve.*' | wc -l | tr -d ' ')
if [ "$n" = "0" ]; then echo "  ok   0 found"; else echo "  $n found"; fail=1; fi

printf '\n\033[1mevery relative link in the top-level docs resolves\033[0m\n'
python3 - <<'PY' || fail=1
import re, pathlib, sys
bad = []
for f in ("README.md", "RESOURCES.md", "LANDSCAPE.md", "TASK_FORMAT.md", "concepts/README.md"):
    p = pathlib.Path(f)
    if not p.is_file():
        continue
    for m in re.finditer(r'\[([^\]]+)\]\((?!https?:|#|mailto:)([^)]+)\)', p.read_text()):
        t = (p.parent / m.group(2).split("#")[0]).resolve()
        if not t.exists():
            bad.append(f"{f}: {m.group(2)}")
print("  " + ("ok   none broken" if not bad else "\n  ".join(bad)))
sys.exit(1 if bad else 0)
PY

if [ "${1:-}" = "--full" ]; then
  run "every task discriminates (reference passes, starter fails)" python3 tools/verify_all.py
fi

printf '\n'
if [ "$fail" = "0" ]; then printf '\033[32mall checks pass\033[0m\n'; else printf '\033[31msomething failed above\033[0m\n'; fi
exit $fail
