def check(workdir):
    from install.matrix import get_version_matrix
    m = {"matrix_complete": 0.0, "architectures_covered": 0.0}
    try:
        mat = get_version_matrix()
    except Exception:
        return m
    if isinstance(mat, dict) and len(mat) >= 3:
        m["matrix_complete"] = 1.0
        m["architectures_covered"] = float(len(mat))
    return m
