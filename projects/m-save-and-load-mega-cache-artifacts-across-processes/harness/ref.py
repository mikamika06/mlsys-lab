import pickle


def save_artifact(path, artifact):
    with open(path, "wb") as f:
        pickle.dump(artifact, f)


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def measure_compile_time(is_warm):
    return 0.1 if is_warm else 5.0


def find_cache_break(base_cfg, new_cfg):
    for k, v in base_cfg.items():
        if new_cfg.get(k) != v:
            return k
    return None
