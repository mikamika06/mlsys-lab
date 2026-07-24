def grade(sol, fx) -> dict:
    """
    Grade the candidate implementation of sdpa_flop_closed_form.
    
    Parameters
    ----------
    sol : module
        Module containing the student's function.
    fx : any
        Unused; present for API compatibility.
    
    Returns
    -------
    dict
        {"exact_match": 1.0} if all tests pass, otherwise {"exact_match": 0.0}.
    """
    # Test cases: (S, d)
    cases = [
        (1, 1),
        (2, 3),
        (5, 10),
        (16, 64),
        (32, 128),
    ]
    
    ok = 1.0
    for S, d in cases:
        try:
            got = sol.sdpa_flop_closed_form(S, d)
        except Exception:
            return {"exact_match": 0.0}
        
        # Reference computed from the closed‑form expression
        ref = 4 * S * S * d
        
        if got != ref or not isinstance(got, int):
            ok = 0.0
            break
    
    return {"exact_match": ok}
