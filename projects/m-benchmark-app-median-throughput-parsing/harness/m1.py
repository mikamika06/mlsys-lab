import ref

def check(workdir):
    from bapp.summary import parse_metrics, compare_hints

    out = {"fps_rel_err": 0.0, "median_rel_err": 0.0}
    cases = ref.get_cases_m1()
    max_fps_err = 0.0
    max_med_err = 0.0
    
    for i, (txt, want) in enumerate(cases):
        try:
            got = parse_metrics(txt)
            fps_err = abs(got["fps"] - want["fps"]) / max(1e-5, want["fps"])
            med_err = abs(got["median"] - want["median"]) / max(1e-5, want["median"])
            max_fps_err = max(max_fps_err, fps_err)
            max_med_err = max(max_med_err, med_err)
        except Exception as e:
            out["_note"] = f"Crash on parse_metrics case {i}: {e}"
            max_fps_err = 1.0
            max_med_err = 1.0
            break

    if max_fps_err < 0.001 and max_med_err < 0.001:
        try:
            lat_log = ref.generate_log(50.0, 20.0)
            tput_log = ref.generate_log(100.0, 40.0)
            comp = compare_hints(lat_log, tput_log)
            expected = {
                "latency_hint_fps": 50.0,
                "latency_hint_median": 20.0,
                "throughput_hint_fps": 100.0,
                "throughput_hint_median": 40.0
            }
            if comp != expected:
                out["_note"] = f"compare_hints mismatch: got {comp}, want {expected}"
                max_fps_err = 1.0
                max_med_err = 1.0
        except Exception as e:
            out["_note"] = f"Crash on compare_hints: {e}"
            max_fps_err = 1.0
            max_med_err = 1.0

    out["fps_rel_err"] = float(max_fps_err)
    out["median_rel_err"] = float(max_med_err)
    return out
