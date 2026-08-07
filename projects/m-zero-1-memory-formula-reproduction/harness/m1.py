import ref


def check(workdir):
    from zero1.memory import calculate_zero1_memory
    from zero1.log_parser import parse_deepspeed_log

    max_rel_err = 0.0
    for tc in ref.TEST_CASES:
        want = ref.calculate_zero1_memory(
            tc["num_params"], tc["world_size"], tc["precision_bytes"]
        )
        got = calculate_zero1_memory(
            tc["num_params"], tc["world_size"], tc["precision_bytes"]
        )

        for key in ["baseline_bytes", "zero1_bytes", "opt_state_per_rank_bytes"]:
            w_val = want[key]
            g_val = float(got.get(key, 0.0))
            err = abs(w_val - g_val) / max(w_val, 1.0)
            if err > max_rel_err:
                max_rel_err = err

    logs_parsed = 1.0
    for log_text, want_parsed in ref.LOG_SAMPLES:
        got_parsed = parse_deepspeed_log(log_text)
        if (
            got_parsed.get("zero_stage") != want_parsed["zero_stage"]
            or got_parsed.get("optimizer_partitioned")
            != want_parsed["optimizer_partitioned"]
            or abs(
                got_parsed.get("estimated_reduction", 0.0)
                - want_parsed["estimated_reduction"]
            )
            > 1e-4
        ):
            logs_parsed = 0.0
            break

    return {"rel_err": float(max_rel_err), "logs_parsed": logs_parsed}
