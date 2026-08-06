"""Checker for Milestone 1: Draft/Target pair compatibility validation."""

import os
import sys


def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "harness"))
    sys.path.insert(0, workdir)
    import ref
    from specdec.pair import is_valid_draft_target_pair

    pairs = ref.generate_pairs()
    matched = 0

    for idx, (draft, target, expected) in enumerate(pairs):
        try:
            res = is_valid_draft_target_pair(draft, target)
            if bool(res) == expected:
                matched += 1
            else:
                return {
                    "pairs_checked": 0.0,
                    "_note": f"Mismatch at pair index {idx}: expected {expected}, got {res}"
                }
        except Exception as e:
            return {
                "pairs_checked": 0.0,
                "_note": f"Exception encountered at index {idx}: {type(e).__name__}: {str(e)}"
            }

    return {"pairs_checked": 1.0 if matched == len(pairs) else 0.0}
