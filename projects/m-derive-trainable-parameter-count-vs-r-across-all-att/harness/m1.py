import sys
from pathlib import Path
import ref


def check(workdir):
    sys.path.insert(0, str(Path(workdir) / "reference"))
    sys.path.insert(0, str(workdir))

    from loraspec.params import count_trainable_parameters
    from loraspec.resolver import resolve_target_modules

    out = {"params_matched": 0.0, "resolved_matched": 0.0}

    targets_list = [
        ["q_proj", "v_proj"],
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    ]

    params_ok = True
    for cfg in ref.CONFIGS:
        for targets in targets_list:
            for r in [4, 8, 16, 64]:
                want = ref.count_trainable_parameters(cfg, targets, r)
                got = count_trainable_parameters(cfg, targets, r)
                if want != got:
                    params_ok = False
                    out["_note"] = f"Param mismatch: expected {want}, got {got}"
                    break
            if not params_ok:
                break
        if not params_ok:
            break

    if params_ok:
        out["params_matched"] = 1.0

    resolved_ok = True
    for tree in ref.MODULE_TREES:
        for shorthands in [["q_proj", "v_proj"], ["gate_proj", "down_proj"]]:
            want = ref.resolve_target_modules(tree, shorthands)
            got = resolve_target_modules(tree, shorthands)
            if want != got:
                resolved_ok = False
                out["_note"] = f"Resolver mismatch: expected {want}, got {got}"
                break
        if not resolved_ok:
            break

    if resolved_ok:
        out["resolved_matched"] = 1.0

    return out
