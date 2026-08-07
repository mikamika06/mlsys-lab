def diff_responses(native_resp, shim_resp):
    """Diff native and shim response structures."""
    raise NotImplementedError


def recover_timings(event_stream):
    """Recover per-phase timings from event stream timestamps."""
    raise NotImplementedError


def find_ignored_parameter(runner_func, base_params, param_candidates):
    """Find parameter silently ignored by shim."""
    raise NotImplementedError
