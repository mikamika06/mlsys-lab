"""Cache key generation and argument classification."""


def classify_arg(name, value, is_constexpr=False, is_ptr=False):
    """Classifies an argument into its JIT cache role."""
    raise NotImplementedError


def build_cache_key(fn_name, sig_spec, args_kw):
    """Builds a deterministic string key from function spec and runtime arguments."""
    raise NotImplementedError
