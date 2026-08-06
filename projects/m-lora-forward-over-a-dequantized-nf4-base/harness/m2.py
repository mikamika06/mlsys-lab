import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from qlora.accounting import count_trainable_parameters
        from qlora.config import fix_qlora_config
    except ImportError as e:
        return {"accounting_matched": 0, "config_repaired": 0, "_note": f"Import error: {e}"}

    accounting_ok = 1
    target_specs = [["q_proj", "v_proj"], ["mlp"], ["self_attn"]]

    for targets in target_specs:
        want_acc = ref.count_trainable_parameters(ref.SYNTHETIC_LAYOUT, targets, lora_r=8)
        try:
            got_acc = count_trainable_parameters(ref.SYNTHETIC_LAYOUT, targets, lora_r=8)
            if got_acc != want_acc:
                accounting_ok = 0
                note = f"Accounting mismatch for {targets}: got {got_acc}, expected {want_acc}"
                break
        except Exception as e:
            accounting_ok = 0
            note = f"Accounting exception: {e}"
            break

    if not accounting_ok:
        return {"accounting_matched": 0, "config_repaired": 0, "_note": note}

    config_ok = 1
    for cfg in ref.CONFIG_TEST_CASES:
        want_cfg = ref.fix_qlora_config(cfg)
        try:
            got_cfg = fix_qlora_config(cfg)
            if got_cfg != want_cfg:
                config_ok = 0
                note = f"Config mismatch: got {got_cfg}, expected {want_cfg}"
                break
        except Exception as e:
            config_ok = 0
            note = f"Config exception: {e}"
            break

    return {"accounting_matched": float(accounting_ok), "config_repaired": float(config_ok)}
