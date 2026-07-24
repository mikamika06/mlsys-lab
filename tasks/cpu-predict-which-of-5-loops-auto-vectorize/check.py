def grade(sol, fx) -> dict:
    # Reference: static analysis of each loop's vectorizability
    # Loop 0: a[i] = b[i] + c[i]  -> no dep, uniform stride -> True
    # Loop 1: a[i] = a[i-1] + b[i] -> loop-carried dep -> False
    # Loop 2: s += a[i]            -> simple reduction -> True
    # Loop 3: a[i] = b[i] if b[i]>0 else 0 -> branch-free SIMD select -> True
    # Loop 4: a[i] = b[i*i%N]     -> non-uniform stride -> False
    ref = [True, False, True, True, False]

    try:
        result = list(sol.classify_loops())
    except Exception:
        return {"exact_match": 0.0}

    if len(result) != 5:
        return {"exact_match": 0.0}

    matches = sum(1 for r, e in zip(result, ref) if bool(r) == bool(e))
    exact_match = 1.0 if matches == 5 else 0.0
    return {"exact_match": exact_match}
