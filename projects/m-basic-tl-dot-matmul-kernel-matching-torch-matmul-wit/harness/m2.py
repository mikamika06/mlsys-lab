import ref


def check(workdir):
    from triton_matmul.constraints import check_tile_alignment
    from triton_matmul.perf import analyze_tflops

    ok_align = check_tile_alignment(32, 32, 32) is True and check_tile_alignment(15, 32, 32) is False
    fixtures = ref.get_benchmark_fixtures()
    perf_res = analyze_tflops(fixtures, 64, 64, 64)
    ok_perf = isinstance(perf_res, dict) and "tflops" in perf_res and "ratio" in perf_res

    passed = 1.0 if (ok_align and ok_perf) else 0.0
    return {"constraints_matched": passed}
