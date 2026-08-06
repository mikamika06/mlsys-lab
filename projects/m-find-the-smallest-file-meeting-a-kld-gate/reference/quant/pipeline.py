from quant.gate import find_smallest_file
from quant.override import override_kv_config


def run_pipeline(candidates, max_kld, overrides):
    best = find_smallest_file(candidates, max_kld)
    if best is None:
        return None
    updated_config = override_kv_config(best.get("config", {}), overrides)
    res = dict(best)
    res["config"] = updated_config
    return res
