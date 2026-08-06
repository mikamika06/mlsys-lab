"""Option precedence and classification logic."""


def compute_effective_options(modelfile_params, request_options, env_vars):
    """Compute effective options enforcing request > Modelfile > env precedence."""
    raise NotImplementedError


def classify_options(options, executor_fn):
    """Partition options into load-time and sample-time by checking load_duration."""
    raise NotImplementedError
