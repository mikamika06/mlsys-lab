#!/bin/bash
# claim_next_native.sh — atomically claim one NATIVE spec (real C++ / real CUDA).
# Sources, in priority order:
#   1. tools/specs_rebuild/  — python-sim tasks that must be ported to real code
#   2. tools/specs_failed/   — never-built cpu-*/cpp-*/gpu-* specs
# Prints "<id> <action>"; exit 1 when both queues are dry.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
mkdir -p tools/specs_claimed

for f in tools/specs_rebuild/*.spec; do
  [ -e "$f" ] || break
  b=$(basename "$f" .spec)
  action=$(python3 -c "
import json,sys
m={x['id']:x['action'] for x in json.load(open('tools/specs_rebuild/_manifest.json'))}
print(m.get('$b','port-to-real-cpp'))" 2>/dev/null)
  if mv "$f" tools/specs_claimed/ 2>/dev/null; then echo "$b $action"; exit 0; fi
done

for f in tools/specs_failed/*.spec; do
  [ -e "$f" ] || break
  b=$(basename "$f" .spec)
  case "$b" in
    gpu-*) action=build-cu ;;
    cpp-*|cpu-*) action=build-cpp ;;
    *) continue ;;
  esac
  if mv "$f" tools/specs_claimed/ 2>/dev/null; then echo "$b $action"; exit 0; fi
done
exit 1
