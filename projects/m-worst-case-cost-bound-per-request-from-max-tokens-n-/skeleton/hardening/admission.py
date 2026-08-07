"""Admission control based on bounded request costs."""


def admit_request(request_config: dict, profile_params: dict, max_gpu_seconds: float) -> tuple[bool, float, str]:
    """Decide whether to admit a request based on cost limit."""
    raise NotImplementedError
