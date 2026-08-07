import ref


def check(workdir):
    from trtllm_config.translator import translate_vllm_command

    out = {"configs_matched": 0.0}
    ok = 0
    for cmd in ref.CONFIGS:
        want = ref.translate_vllm_command(cmd)
        got = translate_vllm_command(cmd)
        if got == want:
            ok += 1
    out["configs_matched"] = float(ok)
    return out
