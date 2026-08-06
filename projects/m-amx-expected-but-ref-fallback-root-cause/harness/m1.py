import ref

def check(workdir):
    from onednn_diag.fallback import analyze_fallback_causes

    logs = ref.generate_verbose_logs(seed=101)
    # Add explicit reference lines
    logs.extend([
        "onednn,exec,cpu,convolution,ref:any,forward,mb1ic32_unaligned,layout_fail,12.5",
        "onednn,exec,cpu,matmul,reference:gemm,forward,m64n64k64,dt_mismatch,18.2",
        "onednn,exec,cpu,reorder,ref:any,eval,shape_xyz,none,5.1"
    ])

    parsed = analyze_fallback_causes(logs)
    fallbacks_count = len(parsed)

    ok = 0
    for item in parsed:
        if "primitive" in item and "implementation" in item and "reason" in item:
            if "ref" in item["implementation"].lower():
                ok += 1

    return {
        "fallbacks_identified": float(ok)
    }
