def check(workdir):
    import ref
    from runner import adapter

    m = {"matrix_complete": 0.0}
    try:
        mat = adapter.get_compatibility_matrix()
        ref_mat = ref.sample_matrix()
        if isinstance(mat, dict) and all(k in mat for k in ref_mat):
            m["matrix_complete"] = 1.0
    except Exception:
        pass
    return m
