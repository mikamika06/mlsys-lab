def classify_snippet(snippet: str) -> str:
    """Classify code snippet portability tier as 'cuda', 'rocm', or 'portable'."""
    s = snippet.lower()
    cuda_keys = ["ptx", "mbarrier", "wgmma", "cp.async", "%sm_", "nvvm", "cubin"]
    rocm_keys = ["amdgcn", "s_waitcnt", "v_mfma", "ds_read", "rocwmma", "gcn"]

    has_cuda = any(k in s for k in cuda_keys)
    has_rocm = any(k in s for k in rocm_keys)

    if has_cuda and not has_rocm:
        return "cuda"
    if has_rocm and not has_cuda:
        return "rocm"
    return "portable"
