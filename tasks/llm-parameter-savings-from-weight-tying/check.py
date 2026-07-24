def grade(sol, fx) -> dict:
    """
    Grade a candidate solution for param_savings.
    
    Parameters
    ----------
    sol : module
        The student's module containing the function to test.
    fx : any
        Unused; present for API compatibility.
    
    Returns
    -------
    dict
        Mapping from metric name to computed value.  Only 'size_ratio' is used.
    """
    cases = [
        (10, 3),
        (1000, 768),
        (50000, 128),
        (1_000_000, 64),   # large case to ensure integer arithmetic works
    ]
    
    ok = True
    for vocab_size, d_model in cases:
        try:
            got = sol.param_savings(vocab_size, d_model)
        except Exception as e:
            return {"size_ratio": 0.0}
        
        if not isinstance(got, (tuple, list)) or len(got) != 2:
            ok = False
            break
        
        tied, untied = got
        expected_tied = vocab_size * d_model
        expected_untied = 2 * vocab_size * d_model
        
        if tied != expected_tied or untied != expected_untied:
            ok = False
            break
    
    ratio = (expected_untied / expected_tied) if ok else 0.0
    return {"size_ratio": float(ratio)}
