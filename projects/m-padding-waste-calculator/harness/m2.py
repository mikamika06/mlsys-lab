import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from varpack.packing import pack_sequences_ffd, compute_packing_efficiency
    from varpack.offsets import build_cu_seqlens, build_sequence_metadata

    out = {"packing_matches": 0.0, "offsets_match": 0.0}

    pack_ok = True
    offset_ok = True

    for ds, cap in zip(ref.DATASETS, ref.MAX_CAPACITIES):
        want_bins = ref.ref_pack_sequences_ffd(ds, cap)
        got_bins = pack_sequences_ffd(ds, cap)

        want_eff = ref.ref_compute_packing_efficiency(want_bins, cap)
        got_eff = compute_packing_efficiency(got_bins, cap)

        if not np.isclose(want_eff, got_eff, rtol=1e-5, atol=1e-5):
            pack_ok = False
            out["_note"] = f"Efficiency mismatch: got {got_eff}, want {want_eff}"
            break

        want_cu = ref.ref_build_cu_seqlens(want_bins)
        got_cu = build_cu_seqlens(got_bins)

        if not np.array_equal(want_cu, got_cu):
            offset_ok = False
            out["_note"] = f"cu_seqlens mismatch: got {got_cu}, want {want_cu}"
            break

        want_meta = ref.ref_build_sequence_metadata(want_bins)
        got_meta = build_sequence_metadata(got_bins)

        if want_meta != got_meta:
            offset_ok = False
            out["_note"] = f"Metadata mismatch: got {got_meta}, want {want_meta}"
            break

    if pack_ok:
        out["packing_matches"] = 1.0
    if offset_ok:
        out["offsets_match"] = 1.0

    return out
