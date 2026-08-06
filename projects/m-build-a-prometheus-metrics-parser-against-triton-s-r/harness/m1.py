import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness.ref as ref


def check(workdir):
    from triton_metrics.parser import parse_prometheus_text

    text = ref.generate_prometheus_payload(seed=101)
    ref_samples = ref.parse_prometheus_text(text)
    got_samples = parse_prometheus_text(text)

    out = {"samples_parsed": 0.0}

    if len(ref_samples) != len(got_samples):
        out["_note"] = f"Sample count mismatch: expected {len(ref_samples)}, got {len(got_samples)}"
        return out

    matches = 0
    for r, g in zip(ref_samples, got_samples):
        if r.name == g.name and r.labels == g.labels and abs(r.value - g.value) < 1e-5:
            matches += 1

    if matches == len(ref_samples):
        out["samples_parsed"] = 1.0
    else:
        out["_note"] = f"Matched {matches}/{len(ref_samples)} samples exactly."

    return out
