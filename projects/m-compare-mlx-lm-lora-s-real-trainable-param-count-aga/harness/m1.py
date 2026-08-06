import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import harness.ref as ref


def check(workdir):
    out = {"params_matched": 0.0}
    try:
        from mlx_lora_audit.params import (
            audit_param_counts,
            compute_formula_params,
            count_real_trainable_params,
        )

        data = ref.generate_audit_fixture(42)

        formula_lora = compute_formula_params(data["specs"], data["r"], use_dora=False)
        real_lora = count_real_trainable_params(data["layers_lora"])
        res_lora = audit_param_counts(data["specs"], data["r"], data["layers_lora"], use_dora=False)

        formula_dora = compute_formula_params(data["specs"], data["r"], use_dora=True)
        real_dora = count_real_trainable_params(data["layers_dora"])
        res_dora = audit_param_counts(data["specs"], data["r"], data["layers_dora"], use_dora=True)

        lora_ok = (formula_lora == real_lora) and res_lora["matches"]
        dora_ok = (formula_dora == real_dora) and res_dora["matches"]

        if lora_ok and dora_ok:
            out["params_matched"] = 1.0
        else:
            out["_note"] = f"lora_ok={lora_ok}, dora_ok={dora_ok}"
    except Exception as e:  # noqa: BLE001
        out["_note"] = f"Error in m1 check: {type(e).__name__}: {str(e)}"
    return out
