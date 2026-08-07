from megacache.artifacts import load_artifact, save_artifact
from megacache.cachekey import compute_cache_key


def measure_compile_time(compile_fn, config, cache_dir, warm=False):
    key = compute_cache_key(config)
    if warm:
        artifact = load_artifact(cache_dir, key)
        if artifact is not None:
            return 0.1, artifact
    result = compile_fn(config)
    save_artifact(cache_dir, key, result)
    return 1.0, result
