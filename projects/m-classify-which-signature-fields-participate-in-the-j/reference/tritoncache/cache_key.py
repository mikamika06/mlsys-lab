"""Cache key generation and argument classification."""


def classify_arg(name, value, is_constexpr=False, is_ptr=False):
    """Classifies an argument into its JIT cache role."""
    if is_constexpr:
        return ("constexpr", name, repr(value))
    elif is_ptr:
        dtype = getattr(value, "dtype", "ptr")
        stride = getattr(value, "stride", None)
        shape = getattr(value, "shape", None)
        return ("tensor", name, str(dtype), tuple(shape) if shape is not None else (), tuple(stride) if stride is not None else ())
    else:
        dtype = type(value).__name__
        return ("scalar", name, dtype)


def build_cache_key(fn_name, sig_spec, args_kw):
    """Builds a deterministic string key from function spec and runtime arguments."""
    parts = [f"fn:{fn_name}"]
    for param_name, meta in sig_spec.items():
        is_constexpr = meta.get("is_constexpr", False)
        is_ptr = meta.get("is_ptr", False)
        val = args_kw[param_name]
        classified = classify_arg(param_name, val, is_constexpr=is_constexpr, is_ptr=is_ptr)
        parts.append(f"{param_name}:{classified}")
    return "|".join(parts)
