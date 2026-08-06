import ref


def check(workdir):
    from hf_attn.resolver import get_valid_backends
    out = {"exact_match": 0.0}
    configs = [
        ({"is_decoder": True}, {"has_flash": True, "torch_version": 2.2, "dtype": "float16"}),
        ({"is_decoder": False}, {"has_flash": True, "torch_version": 2.2, "dtype": "float16"}),
        ({"is_decoder": True}, {"has_flash": False, "torch_version": 2.2, "dtype": "bfloat16"}),
        ({"is_decoder": True}, {"has_flash": True, "torch_version": 1.9, "dtype": "float32"}),
        ({"is_decoder": True}, {"has_flash": True, "torch_version": 2.1, "dtype": "bfloat16"}),
    ]
    ok = 0
    for c, e in configs:
        want = ref.get_valid_backends(c, e)
        try:
            got = get_valid_backends(c, e)
        except Exception:
            continue
        if want == got:
            ok += 1
    if ok == len(configs):
        out["exact_match"] = 1.0
    return out
