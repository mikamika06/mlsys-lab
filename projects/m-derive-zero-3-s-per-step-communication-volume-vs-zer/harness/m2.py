import ref


def check(workdir):
    from zeroutil.logs import parse_deepspeed_init_log, parse_zero_runtime_log

    out = {"logs_parsed": 0.0, "memory_reduction_match": 0.0}
    parsed_init = parse_deepspeed_init_log(ref.INIT_LOGS[0])
    parsed_run = parse_zero_runtime_log(ref.RUNTIME_LOGS[0])

    if isinstance(parsed_init, dict) and parsed_init.get(0) == 250000:
        out["logs_parsed"] = 1.0
    if isinstance(parsed_run, dict) and abs(parsed_run.get(0, 0.0) - 3.5) < 1e-5:
        out["memory_reduction_match"] = 1.0
    return out
