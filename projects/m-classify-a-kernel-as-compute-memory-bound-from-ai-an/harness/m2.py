import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    import roofline.intensity as intensity

    out = {"intensity_matched": 0.0, "rankings_matched": 0.0}

    ai_ok = True
    for item in ref.KERNEL_SHAPES:
        kind = item["kind"]
        dt = item["dtype_bytes"]
        if kind == "vector_add":
            want = ref.compute_vector_add_ai(item["n"], dt)
            got = intensity.compute_vector_add_ai(item["n"], dt)
        elif kind == "gemv":
            want = ref.compute_gemv_ai(item["m"], item["n"], dt)
            got = intensity.compute_gemv_ai(item["m"], item["n"], dt)
        elif kind == "gemm":
            want = ref.compute_gemm_ai(item["m"], item["n"], item["k"], dt)
            got = intensity.compute_gemm_ai(item["m"], item["n"], item["k"], dt)
        elif kind == "bmm":
            want = ref.compute_bmm_ai(item["b"], item["m"], item["n"], item["k"], dt)
            got = intensity.compute_bmm_ai(item["b"], item["m"], item["n"], item["k"], dt)
        elif kind == "conv2d":
            want = ref.compute_conv2d_ai(item["n"], item["c_in"], item["c_out"], item["h"], item["w"], item["k"], dt)
            got = intensity.compute_conv2d_ai(item["n"], item["c_in"], item["c_out"], item["h"], item["w"], item["k"], dt)
        elif kind == "layernorm":
            want = ref.compute_layernorm_ai(item["b"], item["s"], item["d"], dt)
            got = intensity.compute_layernorm_ai(item["b"], item["s"], item["d"], dt)

        if abs(got - want) > 1e-4:
            ai_ok = False
            out["_note"] = f"AI mismatch for {kind}: got {got}, want {want}"
            break

    want_rank = ref.rank_kernels_by_intensity(ref.RANKING_TEST_SET)
    got_rank = intensity.rank_kernels_by_intensity(ref.RANKING_TEST_SET)

    rank_ok = got_rank == want_rank
    if not rank_ok and "_note" not in out:
        out["_note"] = f"Ranking got {got_rank}, want {want_rank}"

    out["intensity_matched"] = 1.0 if ai_ok else 0.0
    out["rankings_matched"] = 1.0 if rank_ok else 0.0

    return out
