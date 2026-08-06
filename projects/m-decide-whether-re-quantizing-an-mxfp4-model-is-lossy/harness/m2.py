import ref


def check(workdir):
    from mxfp4.moe import compute_mxfp4_share

    out = {"share_matched": 0.0}
    ok = 0
    for spec in ref.SPECS:
        want = 0.0
        total_b = 0
        mxfp_b = 0
        for layer in spec.get("layers", []):
            if layer.get("type") == "moe":
                ne = layer.get("num_experts", 1)
                ep = layer.get("expert_params", 0)
                rp = layer.get("router_params", 0)
                bs = layer.get("block_size", 32)
                eb = (ep * 4) // 8 + (ep // bs) * 1
                te = eb * ne
                tr = rp * 2
                mxfp_b += te
                total_b += te + tr
            else:
                p = layer.get("params", 0)
                total_b += p * 2
        want = float(mxfp_b) / float(total_b) if total_b > 0 else 0.0
        got = compute_mxfp4_share(spec)
        if abs(got - want) < 1e-5:
            ok += 1
    if ok == len(ref.SPECS):
        out["share_matched"] = 1.0
    return out
