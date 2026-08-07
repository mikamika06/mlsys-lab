import ref


def check(workdir):
    out = {"verified_matches": 0.0}
    try:
        from peft_verify.verify import check_modules_to_save_trainable

        model_good = ref.build_test_model(freeze_backbone=True, train_head=True)
        res_good = check_modules_to_save_trainable(model_good, ["classifier"])

        model_bad_base = ref.build_test_model(freeze_backbone=False, train_head=True)
        res_bad_base = check_modules_to_save_trainable(model_bad_base, ["classifier"])

        model_bad_head = ref.build_test_model(freeze_backbone=True, train_head=False)
        res_bad_head = check_modules_to_save_trainable(model_bad_head, ["classifier"])

        if (
            res_good.get("valid") is True
            and res_good.get("base_frozen") is True
            and res_good.get("head_trainable") is True
            and res_bad_base.get("base_frozen") is False
            and res_bad_base.get("valid") is False
            and res_bad_head.get("head_trainable") is False
            and res_bad_head.get("valid") is False
        ):
            out["verified_matches"] = 1.0
        else:
            out["_note"] = f"mismatch: good={res_good}, bad_base={res_bad_base}, bad_head={res_bad_head}"
    except Exception as e:
        out["_note"] = f"exception: {type(e).__name__}: {str(e)[:120]}"
    return out
