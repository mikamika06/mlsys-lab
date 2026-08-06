import ref


def check(workdir):
    from ggufmap.unmapped import find_unmapped

    _, (hf_keys, mapped, expected), _ = ref.get_cases()
    try:
        res = find_unmapped(hf_keys, mapped)
        match = res == expected
    except Exception:
        match = False
    return {"unmapped_matched": 1.0 if match else 0.0}
