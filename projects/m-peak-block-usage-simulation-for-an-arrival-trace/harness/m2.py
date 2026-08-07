import ref


def check(workdir):
    from kvtrace.analysis import compute_contiguous_waste, compute_paged_waste, compute_waste_ratio

    out = {
        "paged_waste_matched": 0.0,
        "contiguous_waste_matched": 0.0,
        "waste_ratio_matched": 0.0,
    }

    paged_ok = True
    cont_ok = True
    ratio_ok = True

    for i, (hist, block_size, max_len) in enumerate(ref.HISTOGRAMS):
        want_paged = ref.compute_paged_waste(hist, block_size)
        want_cont = ref.compute_contiguous_waste(hist, max_len)
        want_ratio = ref.compute_waste_ratio(hist, block_size, max_len)

        try:
            got_paged = compute_paged_waste(hist, block_size)
            if got_paged != want_paged:
                paged_ok = False
                if "_note" not in out:
                    out["_note"] = f"paged waste mismatch on hist {i}: got {got_paged}, expected {want_paged}"
        except Exception as e:  # noqa: BLE001
            paged_ok = False
            if "_note" not in out:
                out["_note"] = f"paged waste raised error on hist {i}: {str(e)[:100]}"

        try:
            got_cont = compute_contiguous_waste(hist, max_len)
            if got_cont != want_cont:
                cont_ok = False
                if "_note" not in out:
                    out["_note"] = f"contiguous waste mismatch on hist {i}: got {got_cont}, expected {want_cont}"
        except Exception as e:  # noqa: BLE001
            cont_ok = False
            if "_note" not in out:
                out["_note"] = f"contiguous waste raised error on hist {i}: {str(e)[:100]}"

        try:
            got_ratio = compute_waste_ratio(hist, block_size, max_len)
            if abs(got_ratio - want_ratio) > 1e-5:
                ratio_ok = False
                if "_note" not in out:
                    out["_note"] = f"waste ratio mismatch on hist {i}: got {got_ratio}, expected {want_ratio}"
        except Exception as e:  # noqa: BLE001
            ratio_ok = False
            if "_note" not in out:
                out["_note"] = f"waste ratio raised error on hist {i}: {str(e)[:100]}"

    out["paged_waste_matched"] = 1.0 if paged_ok else 0.0
    out["contiguous_waste_matched"] = 1.0 if cont_ok else 0.0
    out["waste_ratio_matched"] = 1.0 if ratio_ok else 0.0

    return out
