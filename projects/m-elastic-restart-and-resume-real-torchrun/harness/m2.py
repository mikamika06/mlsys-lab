import ref

def check(workdir):
    from elastic.rendezvous import compute_membership
    out = {"membership_matched": 0.0}
    ok = 0
    for old_ws, failed, new_ws, expected in ref.MEMBERSHIP_TESTS:
        got = compute_membership(old_ws, failed, new_ws)
        if got == expected:
            ok += 1
        else:
            out["_note"] = f"membership({old_ws}, {failed}, {new_ws}) got {got}, want {expected}"
            break
    if ok == len(ref.MEMBERSHIP_TESTS):
        out["membership_matched"] = 1.0
    return out
