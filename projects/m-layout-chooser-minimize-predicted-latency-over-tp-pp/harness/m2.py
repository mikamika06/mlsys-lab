import ref


def check(workdir):
    from layout.chooser import select_layout

    out = {"layout_argmins_matched": 0.0}

    ok = True
    for cfg in ref.CONFIGS:
        for vram in ref.VRAM_TESTS:
            for lat_table in ref.LATENCY_TABLES:
                want = ref.select_layout(cfg, vram, lat_table)
                got = select_layout(cfg, vram, lat_table)
                if want != got:
                    ok = False
                    out["_note"] = f"select_layout mismatch: want index {want}, got {got}"
                    break
            if not ok:
                break
        if not ok:
            break

    if ok:
        out["layout_argmins_matched"] = 1.0

    return out
