import numpy as np
import ref


def check(workdir):
    from quant.diagnose import diagnose_noop_layers
    from quant.recover import recover_state_dict

    out = {"noop_detected": 0.0, "recovered_exact_match": 0.0}

    ref_noops = ref.diagnose_noop_layers(ref.BEFORE_MODEL_1, ref.AFTER_MODEL_1)
    user_noops = diagnose_noop_layers(ref.BEFORE_MODEL_1, ref.AFTER_MODEL_1)

    if sorted(user_noops or []) == sorted(ref_noops):
        out["noop_detected"] = 1.0
    else:
        out["_note"] = f"noop diagnosis mismatch: expected {ref_noops}, got {user_noops}"

    sd, expected = ref.SAMPLE_STATE_DICT, ref.SAMPLE_EXPECTED_RECOVERED
    user_recovered = recover_state_dict(sd)

    if isinstance(user_recovered, dict) and set(user_recovered.keys()) == set(
        expected.keys()
    ):
        matches = True
        for k, v_exp in expected.items():
            v_got = user_recovered[k]
            if not isinstance(v_got, np.ndarray) or not np.allclose(
                v_got, v_exp, atol=1e-5
            ):
                matches = False
                break
        if matches:
            out["recovered_exact_match"] = 1.0
        elif "_note" not in out:
            out["_note"] = "recovered state_dict arrays do not match expected values"
    elif "_note" not in out:
        out["_note"] = (
            f"recovered keys mismatch: expected {set(expected.keys())}, got {set((user_recovered or {}).keys())}"
        )

    return out
