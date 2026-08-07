import ref


def check(workdir):
    from ckptutils.exactness import verify_gradient_exactness

    model, inputs, strategies = ref.get_test_setup()
    try:
        err = verify_gradient_exactness(model, inputs, strategies[1])
    except Exception as e:
        return {"exactness_score": 0.0, "_note": f"error: {e}"}

    if not isinstance(err, float):
        return {"exactness_score": 0.0, "_note": "did not return a float"}

    score = 1.0 if err < 1e-4 else 0.0
    return {"exactness_score": score}
