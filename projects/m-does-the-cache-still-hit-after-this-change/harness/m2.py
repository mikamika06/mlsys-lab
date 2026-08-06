import os
import ref


def check(workdir):
    from cacheutils.forensics import inspect_cache

    cache_dir = os.path.join(workdir, "dummy_cache")
    os.makedirs(cache_dir, exist_ok=True)
    sub = os.path.join(cache_dir, "subdir")
    os.makedirs(sub, exist_ok=True)

    p1 = os.path.join(cache_dir, "small.bin")
    p2 = os.path.join(sub, "large.bin")

    with open(p1, "wb") as f:
        f.write(b"a" * 10)
    with open(p2, "wb") as f:
        f.write(b"b" * 100)

    want = ref.analyze_cache_directory(cache_dir)
    try:
        got = inspect_cache(cache_dir)
    except Exception as e:
        return {"forensics_matched": 0.0, "_note": f"raised {type(e).__name__}"}

    match = 1 if got == want else 0
    if not match:
        return {"forensics_matched": 0.0, "_note": f"got {got}, want {want}"}
    return {"forensics_matched": 1.0}
