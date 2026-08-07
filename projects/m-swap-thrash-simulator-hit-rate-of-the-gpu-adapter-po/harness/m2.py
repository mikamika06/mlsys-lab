import ref


def check(workdir):
    from multilora.validator import validate_adapters
    adapters = [{"rank": 8, "memory_mb": 100}]
    limits = {"max_rank": 16, "max_memory_mb": 500}
    try:
        got = validate_adapters(adapters, limits)
        want = ref.validate_adapters(adapters, limits)
        match = 1.0 if got == want else 0.0
    except Exception:
        match = 0.0
    return {"flags_match": match}
