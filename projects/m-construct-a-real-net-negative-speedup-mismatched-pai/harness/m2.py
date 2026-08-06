import sys

def check(workdir):
    sys.path.insert(0, workdir)
    import spec_fail.pathology as p
    import spec_fail.metrics as m

    out = {"negative_speedup": 0.0, "lookup_correct": 0.0, "degenerate_loop": 0.0}

    try:
        conf = p.get_net_negative_speedup_config()
        if len(conf) == 5:
            p_arr, q_arr, gamma, t_draft, t_target = conf
            speedup = m.expected_speedup(p_arr, q_arr, gamma, t_draft, t_target)
            if speedup < 1.0 and gamma > 0 and t_draft > 0 and t_target > 0:
                if abs(p_arr.sum() - 1.0) < 1e-4 and abs(q_arr.sum() - 1.0) < 1e-4:
                    out["negative_speedup"] = 1.0
    except NotImplementedError:
        pass

    try:
        seq = [1, 2, 3, 4, 1, 2, 3]
        drafted = p.prompt_lookup_draft(seq, 2)
        if drafted == [4, 1]:
            seq2 = [9, 9, 9, 9]
            if p.prompt_lookup_draft(seq2, 1) == [9]:
                out["lookup_correct"] = 1.0
    except NotImplementedError:
        pass

    try:
        scenario = p.get_degenerate_loop_scenario()
        if len(scenario) == 2:
            seq, gamma = scenario
            drafted = p.prompt_lookup_draft(seq, gamma)
            if len(drafted) == gamma and gamma >= 4:
                is_loop = False
                for c in [1, 2, 3]:
                    if gamma >= c * 2:
                        match = True
                        for i in range(c, len(drafted)):
                            if drafted[i] != drafted[i-c]:
                                match = False
                        if match:
                            is_loop = True
                            break
                if is_loop and len(set(drafted)) > 0:
                    out["degenerate_loop"] = 1.0
    except NotImplementedError:
        pass

    return out
