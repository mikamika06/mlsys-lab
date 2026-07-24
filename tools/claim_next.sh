#!/bin/bash
# Atomically claim the next Python-native failed spec. Prints the task id, or nothing if dry.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
mkdir -p tools/specs_claimed tools/specs_done tools/specs_failed_hard
shopt -s nullglob
for f in tools/specs_failed/*.spec; do
  b=$(basename "$f" .spec)
  case "$b" in cpp-*|gpu-*|cpu-*) continue ;; esac      # native tracks handled separately
  if mv "$f" tools/specs_claimed/ 2>/dev/null; then echo "$b"; exit 0; fi
done
exit 1
