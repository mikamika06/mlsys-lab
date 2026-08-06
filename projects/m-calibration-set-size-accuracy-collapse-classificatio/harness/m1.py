import ref


def check(workdir):
    from quantcollapse.analysis import reconstruct_scales
    nodes, _, _, _ = ref.get_test_data()
    ref_scales = reconstruct_scales(nodes)
    try:
        from quantcollapse.analysis import reconstruct_scales as user_fn
        user_scales = user_fn(nodes)
    except Exception as e:
        return {"scales_matched": 0.0, "_note": f"raised {type(e).__name__}"}
    
    if len(user_scales) != len(ref_scales):
        return {"scales_matched": 0.0, "_note": "length mismatch"}
    
    ok = 1
    for a, b in zip(ref_scales, user_scales):
        if abs(float(a) - float(b)) > 1e-4:
            ok = 0
            break
    return {"scales_matched": float(ok)}
