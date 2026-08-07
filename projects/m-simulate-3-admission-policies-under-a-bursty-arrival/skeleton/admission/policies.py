def token_bucket_policy(arrivals, capacity, refill_rate):
    raise NotImplementedError


def concurrency_limit_policy(arrivals, max_concurrency):
    raise NotImplementedError


def delay_threshold_policy(arrivals, max_delay):
    raise NotImplementedError
