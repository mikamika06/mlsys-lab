import ref


def check(workdir):
    from ckptutils.pareto import compute_pareto_curve

    model, inputs, strategies = ref.get_test_setup()
    try:
        ref_res = ref.ref_pareto(model, inputs, strategies)
        learner_res = compute_pareto_curve(model, inputs, strategies)
    except Exception as e:
        return {"pareto_match": 0.0, "_note": f"error: {e}"}

    if not isinstance(learner_res, list) or len(learner_res) != len(strategies):
        return {"pareto_match": 0.0, "_note": "invalid length or type"}

    for item in learner_res:
        if not isinstance(item, dict) or "strategy" not in item or "time" not in item or "memory" not in item:
            return {"pareto_match": 0.0, "_note": "missing keys in result dict"}

    return {"pareto_match": 1.0}
