"""Milestone 1 harness check."""
import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"layout_matches": 0.0}
    try:
        from tpquant.layout import analyze_tp_slice, build_g_idx
    except ImportError as e:
        out["_note"] = f"ImportError: {e}"
        return out

    ok = 0
    total = len(ref.CONFIGS)
    for cfg in ref.CONFIGS:
        in_feat = cfg["in_features"]
        g_size = cfg["group_size"]
        tp_size = cfg["tp_size"]
        perm = cfg["perm"]

        ref_gidx = ref.build_g_idx(in_feat, g_size, perm)
        got_gidx = build_g_idx(in_feat, g_size, perm)

        ref_analysis = ref.analyze_tp_slice(in_feat, g_size, tp_size, perm)
        got_analysis = analyze_tp_slice(in_feat, g_size, tp_size, perm)

        gidx_ok = (ref_gidx == got_gidx).all()
        analysis_ok = (
            got_analysis.get("is_safe") == ref_analysis["is_safe"]
            and got_analysis.get("oob_ranks") == ref_analysis["oob_ranks"]
            and got_analysis.get("fragmented_groups_count") == ref_analysis["fragmented_groups_count"]
        )

        if gidx_ok and analysis_ok:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"mismatch on config: {cfg}"

    if ok == total:
        out["layout_matches"] = 1.0

    return out
