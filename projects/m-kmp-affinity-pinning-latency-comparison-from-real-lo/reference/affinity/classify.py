def classify_subscription(config):
    threads = config.get("threads", 1)
    cores = config.get("cores", 1)
    if threads > cores:
        return "oversubscribed"
    elif threads < cores:
        return "under-pinned"
    else:
        return "optimal"
