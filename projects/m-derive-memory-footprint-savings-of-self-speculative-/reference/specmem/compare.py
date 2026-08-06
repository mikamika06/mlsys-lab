from specmem.footprint import separate_draft_footprint, self_speculative_footprint


def compute_savings(target_config, draft_config, extra_config, batch_size, seq_len):
    sep = separate_draft_footprint(target_config, draft_config, batch_size, seq_len)
    self_s = self_speculative_footprint(target_config, extra_config, batch_size, seq_len)
    saved_bytes = sep["total_bytes"] - self_s["total_bytes"]
    savings_fraction = saved_bytes / float(sep["total_bytes"]) if sep["total_bytes"] > 0 else 0.0
    return {
        "separate_total": sep["total_bytes"],
        "self_total": self_s["total_bytes"],
        "saved_bytes": saved_bytes,
        "savings_fraction": savings_fraction
    }
