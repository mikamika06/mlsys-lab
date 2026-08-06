import ref

def check(workdir):
    from profdebug.permissions import predict_perm
    ok = 0
    for t in ref.PERM_TESTS:
        got = predict_perm(t["regkey"], t["groups"], t["root"])
        if got == t["expect"]:
            ok += 1
    match = 1.0 if ok == len(ref.PERM_TESTS) else 0.0
    return {"prediction_match": match}
