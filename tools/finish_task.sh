#!/bin/bash
# finish_task.sh <id> ok|fail  — move the claimed spec to done/ or failed_hard/
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
id="$1"; r="$2"
[ "$r" = "ok" ] && mv "tools/specs_claimed/$id.spec" tools/specs_done/ 2>/dev/null \
                || mv "tools/specs_claimed/$id.spec" tools/specs_failed_hard/ 2>/dev/null
