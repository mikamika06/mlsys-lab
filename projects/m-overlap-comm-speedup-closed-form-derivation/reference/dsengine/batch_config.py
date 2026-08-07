def validate_batch_config(config: dict) -> bool:
    required_keys = {
        "train_batch_size",
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "data_parallel_size",
    }
    if not required_keys.issubset(config.keys()):
        return False
    tbs = config["train_batch_size"]
    mbs = config["train_micro_batch_size_per_gpu"]
    gas = config["gradient_accumulation_steps"]
    dp = config["data_parallel_size"]
    if any(x <= 0 for x in (tbs, mbs, gas, dp)):
        return False
    return tbs == mbs * gas * dp


def resolve_batch_config(config: dict) -> dict:
    keys = {
        "train_batch_size",
        "train_micro_batch_size_per_gpu",
        "gradient_accumulation_steps",
        "data_parallel_size",
    }
    provided = {k: config[k] for k in keys if k in config and config[k] is not None}
    if len(provided) < 3:
        raise ValueError("At least 3 batch parameters must be provided")

    res = dict(provided)
    if len(provided) == 4:
        if not validate_batch_config(res):
            raise ValueError("Inconsistent batch configuration parameters")
        return res

    tbs = res.get("train_batch_size")
    mbs = res.get("train_micro_batch_size_per_gpu")
    gas = res.get("gradient_accumulation_steps")
    dp = res.get("data_parallel_size")

    if tbs is None:
        res["train_batch_size"] = mbs * gas * dp
    elif mbs is None:
        denom = gas * dp
        if tbs % denom != 0:
            raise ValueError("train_batch_size is not divisible by gas * dp")
        res["train_micro_batch_size_per_gpu"] = tbs // denom
    elif gas is None:
        denom = mbs * dp
        if tbs % denom != 0:
            raise ValueError("train_batch_size is not divisible by mbs * dp")
        res["gradient_accumulation_steps"] = tbs // denom
    elif dp is None:
        denom = mbs * gas
        if tbs % denom != 0:
            raise ValueError("train_batch_size is not divisible by mbs * gas")
        res["data_parallel_size"] = tbs // denom

    if not validate_batch_config(res):
        raise ValueError("Resolved configuration is invalid")

    return res
