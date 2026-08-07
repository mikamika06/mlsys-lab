from reference.admission.policies import token_bucket_policy


def simulate_trace(arrivals, policy_func, *args, **kwargs):
    return policy_func(arrivals, *args, **kwargs)
