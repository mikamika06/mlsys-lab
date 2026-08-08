def check(workdir):
    m = {"accuracy_target_met": 0.0, "speedup_target_met": 0.0}
    try:
        from int8_eng.tuning import evaluate_engine
        model_int8 = {"calibrated": True}
        eval_res = evaluate_engine(model_int8, [1.0, 2.0])
        if eval_res.get("accuracy", 0.0) >= 0.97:
            m["accuracy_target_met"] = 1.0
        if eval_res.get("speedup", 0.0) >= 0.70:
            m["speedup_target_met"] = 1.0
    except Exception:
        pass
    return m
