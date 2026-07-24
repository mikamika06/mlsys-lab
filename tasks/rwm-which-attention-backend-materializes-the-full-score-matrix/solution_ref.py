def classify_backend(name: str) -> str:
    """
    Return 'full' if the backend materialises the full N×N score matrix,
    otherwise return 'efficient'.
    """
    mapping = {
        "naive": "full",
        "math-SDPA": "full",
        "mem-efficient": "efficient",
        "flash": "efficient"
    }
    return mapping[name]
