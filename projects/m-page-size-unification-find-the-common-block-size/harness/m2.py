import ref


def check(workdir):
    try:
        from kvpage.alloc import allocate_pages
        from kvpage.metrics import total_memory_and_waste
    except Exception as e:
        return {"alloc_matched": 0.0, "metrics_matched": 0.0, "_note": f"Import error: {e}"}

    out = {"alloc_matched": 1.0, "metrics_matched": 1.0}

    seq_lens = [1, 15, 33, 128]
    alignments = [64, 128]

    for i, cfg in enumerate(ref.CONFIGS):
        block_size = ref.find_common_block_size(cfg)
        for seq_len in seq_lens:
            for align in alignments:
                want_alloc = ref.allocate_pages(cfg, seq_len, block_size, page_align_bytes=align)
                want_metrics = ref.total_memory_and_waste(cfg, seq_len, block_size, page_align_bytes=align)
                try:
                    got_alloc = allocate_pages(cfg, seq_len, block_size, page_align_bytes=align)
                    if got_alloc != want_alloc:
                        out["alloc_matched"] = 0.0
                        if "_note" not in out:
                            out["_note"] = f"Config {i} alloc mismatch at len={seq_len}: got {got_alloc}, want {want_alloc}"
                except Exception as e:
                    out["alloc_matched"] = 0.0
                    if "_note" not in out:
                        out["_note"] = f"allocate_pages raised {type(e).__name__}: {e}"

                try:
                    got_metrics = total_memory_and_waste(cfg, seq_len, block_size, page_align_bytes=align)
                    if got_metrics != want_metrics:
                        out["metrics_matched"] = 0.0
                        if "_note" not in out:
                            out["_note"] = f"Config {i} metrics mismatch at len={seq_len}: got {got_metrics}, want {want_metrics}"
                except Exception as e:
                    out["metrics_matched"] = 0.0
                    if "_note" not in out:
                        out["_note"] = f"total_memory_and_waste raised {type(e).__name__}: {e}"

    return out
