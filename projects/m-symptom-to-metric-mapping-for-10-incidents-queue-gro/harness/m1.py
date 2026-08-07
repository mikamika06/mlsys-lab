"""Checker for Milestone 1: Incident mapping and telemetry parsing."""

import ref


def check(workdir):
    out = {"incidents_mapped": 0.0, "symptoms_parsed": 0.0}

    try:
        from vllm_obs.mapping import map_incident_to_metric, parse_telemetry_sample
    except ImportError as e:
        out["_note"] = f"Failed to import vllm_obs.mapping: {e}"
        return out

    mapped_ok = 0
    for i in range(1, 11):
        try:
            res = map_incident_to_metric(i)
            if isinstance(res, dict) and "primary_metric" in res and "symptom" in res:
                mapped_ok += 1
        except Exception as e:
            out["_note"] = f"Incident {i} mapping failed: {e}"
            return out

    if mapped_ok == 10:
        out["incidents_mapped"] = 1.0

    parsed_ok = True
    for sample in ref.METRIC_SAMPLES:
        try:
            got = parse_telemetry_sample(sample["raw"])
            exp = sample["expected"]
            for k, want_val in exp.items():
                if got.get(k) != want_val:
                    parsed_ok = False
                    out["_note"] = f"Mismatch for {k}: got {got.get(k)}, expected {want_val}"
                    break
        except Exception as e:
            parsed_ok = False
            out["_note"] = f"parse_telemetry_sample failed: {e}"
            break

    if parsed_ok:
        out["symptoms_parsed"] = 1.0

    return out
