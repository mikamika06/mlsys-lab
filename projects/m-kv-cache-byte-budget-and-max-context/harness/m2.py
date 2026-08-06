import sys
import ref

sys.path.insert(0, ".")


def check(workdir):
    try:
        from exporter.convert import build_toy_decoder_spec, simulate_export
        from exporter.repair import check_state_alignment, repair_state_names
    except Exception as e:
        return {
            "conversion_passed": 0.0,
            "repair_passed": 0.0,
            "_note": f"Import error: {e}",
        }

    out = {"conversion_passed": 0.0, "repair_passed": 0.0}

    try:
        spec = build_toy_decoder_spec(
            num_layers=4,
            hidden_dim=256,
            num_kv_heads=4,
            head_dim=64,
            max_context=1024,
        )
        exp = simulate_export(spec, state_names=("k_state", "v_state"))
        if len(exp.get("description", {}).get("states", [])) == 8:
            out["conversion_passed"] = 1.0
        else:
            out["_note"] = "Converted model specification missing required states"
    except Exception as e:
        out["_note"] = f"Conversion step failed: {e}"
        return out

    try:
        repaired = repair_state_names(exp, ("key_cache", "value_cache"))
        if check_state_alignment(repaired, ("key_cache", "value_cache")):
            out["repair_passed"] = 1.0
        else:
            out["_note"] = "Repair state names failed to align StateType metadata"
    except Exception as e:
        out["_note"] = f"Repair step failed: {e}"

    return out
