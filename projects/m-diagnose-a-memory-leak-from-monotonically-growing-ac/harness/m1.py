import ref

def check(workdir):
    from memdiag.leak import diagnose_leak
    fixtures = ref.get_leak_fixtures()
    match = 0
    for fix in fixtures:
        got = diagnose_leak(fix["snapshots"])
        if got == fix["expected"]:
            match += 1
    return {"leaks_matched": 1.0 if match == len(fixtures) else 0.0}
