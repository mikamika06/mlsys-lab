import ref


def check(workdir):
    from buildfix.jobs import compute_max_jobs
    ok = 0
    for ram, cores, want in ref.TEST_CASES_JOBS:
        got = compute_max_jobs(ram, cores)
        if got == want:
            ok += 1
    return {"jobs_matched": 1.0 if ok == len(ref.TEST_CASES_JOBS) else 0.0}
