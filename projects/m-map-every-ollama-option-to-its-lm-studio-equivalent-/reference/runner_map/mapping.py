MAPPING = {
    "num_ctx": {"equivalent": "context_length", "supported": True},
    "temperature": {"equivalent": "temperature", "supported": True},
    "top_p": {"equivalent": "top_p", "supported": True},
    "num_predict": {"equivalent": "max_tokens", "supported": True},
    "repeat_penalty": {"equivalent": "repeat_penalty", "supported": True},
    "stop": {"equivalent": "stop", "supported": True},
    "seed": {"equivalent": "seed", "supported": True},
    "num_thread": {"equivalent": "cpu_threads", "supported": True},
    "tfs_z": {"equivalent": None, "supported": False},
    "typical_p": {"equivalent": None, "supported": False},
}


def map_option(opt_name):
    return MAPPING.get(opt_name, {"equivalent": None, "supported": False})


def map_config(cfg):
    out = {}
    gaps = []
    for k, v in cfg.items():
        res = map_option(k)
        if res["supported"]:
            out[res["equivalent"]] = v
        else:
            gaps.append(k)
    return {"mapped": out, "gaps": gaps}
