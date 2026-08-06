import ref

def check(workdir):
    from bapp.pc import parse_top_5

    out = {"top5_match": 0.0}
    cases = ref.get_cases_m2()
    ok = 0
    
    for i, (txt, want) in enumerate(cases):
        try:
            got = parse_top_5(txt)
            if got == want:
                ok += 1
            else:
                out["_note"] = f"case {i}: got {got}, want {want}"
        except Exception as e:
            out["_note"] = f"Crash on case {i}: {e}"
            break
            
    out["top5_match"] = 1.0 if ok == len(cases) else 0.0
    return out
