import ref

def check(workdir):
    from recomp.specialization import minimal_specializations
    shapes, budget = ref.get_fixtures()
    want = ref.minimal_specializations(shapes, budget)
    try:
        got = minimal_specializations(shapes, budget)
    except Exception as e:
        return {"specializations_match": 0.0, "_note": f"raised {type(e).__name__}"}
    
    match = (got == want) and (len(got) <= budget)
    return {"specializations_match": 1.0 if match else 0.0, "_note": f"got {got}, want {want}"}
