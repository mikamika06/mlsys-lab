import ref


def check(workdir):
    from quantcollapse.analysis import classify_collapse
    _, sizes, variances, threshold = ref.get_test_data()
    ref_res = classify_collapse(sizes, variances, threshold)
    try:
        from quantcollapse.analysis import classify_collapse as user_fn
        user_res = user_fn(sizes, variances, threshold)
    except Exception as e:
        return {"classification_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    
    if user_res == ref_res:
        return {"classification_matched": 1.0}
    return {"classification_matched": 0.0, "_note": f"got {user_res}, want {ref_res}"}
