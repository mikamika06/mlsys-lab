import ref


def check(workdir):
    from ollama_evict.manager import ModelManager
    out = {"eviction_match": 0.0}
    try:
        ok = True
        notes = []
        for i, scn in enumerate(ref.get_test_scenarios()):
            ref_res = ref.run_reference(scn)
            mgr = ModelManager(scn["max_loaded"])
            got_res = []
            for req in scn["requests"]:
                res = mgr.request(req)
                got_res.append(res)
            if got_res != ref_res:
                ok = False
                notes.append(f"scn {i}: got {got_res}, want {ref_res}")
        out["eviction_match"] = 1.0 if ok else 0.0
        if notes:
            out["_note"] = notes[0]
    except Exception as e:
        out["_note"] = f"error in m2: {type(e).__name__}: {str(e)[:120]}"
    return out
