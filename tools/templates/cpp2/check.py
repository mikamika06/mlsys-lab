from mlsys.sim import abi as cppabi

def grade(sol, fx) -> dict:
    cases = [["char","int","double"], ["char","char"], ["double","char","int"],
             ["short","double"], ["char"], ["int","int","char"]]
    ok = 1.0
    for fields in cases:
        try:
            got = sol.struct_size(list(fields))
        except Exception:
            ok = 0.0; break
        if got != cppabi.sizeof(fields):   # pinned LP64 ABI is the oracle
            ok = 0.0; break
    return {"exact_match": ok}
