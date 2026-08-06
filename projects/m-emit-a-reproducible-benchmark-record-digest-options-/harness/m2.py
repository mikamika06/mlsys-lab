import ref


def check(workdir):
    from benchmark.analysis import cold_start_inflation, identify_flaws

    out = {"flaws_matched": 0.0}
    sample_script = "import time\ndef run():\n    pass"
    got_flaws = identify_flaws(sample_script)
    want_flaws = ref.analyze_flaws(sample_script)
    got_inf = cold_start_inflation(0.5, 1.0, 1000)
    want_inf = ref.quantify_cold_start(0.5, 1.0, 1000)
    if got_flaws == want_flaws and abs(got_inf - want_inf) < 1e-5:
        out["flaws_matched"] = 1.0
    return out
