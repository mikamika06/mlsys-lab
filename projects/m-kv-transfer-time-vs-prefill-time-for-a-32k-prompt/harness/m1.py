import ref


def check(workdir):
    from disagg.transfer import analyze_kv_transfer, kv_cache_bytes, prefill_time_ms, transfer_time_ms

    out = {"transfer_rel_err": 1.0, "prefill_rel_err": 1.0}
    max_trans_err = 0.0
    max_pref_err = 0.0

    for plen, m_cfg, h_cfg in ref.TEST_CONFIGS:
        ref_kb = ref.kv_cache_bytes(
            plen, m_cfg["num_layers"], m_cfg["num_kv_heads"], m_cfg["head_dim"], m_cfg.get("dtype_bytes", 2)
        )
        got_kb = kv_cache_bytes(
            plen, m_cfg["num_layers"], m_cfg["num_kv_heads"], m_cfg["head_dim"], m_cfg.get("dtype_bytes", 2)
        )
        if ref_kb != got_kb:
            out["_note"] = f"kv_cache_bytes mismatch: got {got_kb}, want {ref_kb}"
            return out

        ref_pref = ref.prefill_time_ms(plen, m_cfg, h_cfg["prefill_tflops"])
        got_pref = prefill_time_ms(plen, m_cfg, h_cfg["prefill_tflops"])
        err_p = abs(got_pref - ref_pref) / max(ref_pref, 1e-9)
        if err_p > max_pref_err:
            max_pref_err = err_p

        ref_trans = ref.transfer_time_ms(ref_kb, h_cfg["bandwidth_gbps"], h_cfg.get("latency_ms", 0.0))
        got_trans = transfer_time_ms(got_kb, h_cfg["bandwidth_gbps"], h_cfg.get("latency_ms", 0.0))
        err_t = abs(got_trans - ref_trans) / max(ref_trans, 1e-9)
        if err_t > max_trans_err:
            max_trans_err = err_t

        ref_res = ref.analyze_kv_transfer(plen, m_cfg, h_cfg)
        got_res = analyze_kv_transfer(plen, m_cfg, h_cfg)
        for key in ("kv_bytes", "prefill_ms", "transfer_ms", "ratio"):
            if key not in got_res:
                out["_note"] = f"missing key {key} in analyze_kv_transfer output"
                return out
            err_k = abs(got_res[key] - ref_res[key]) / max(abs(ref_res[key]), 1e-9)
            if err_k > 1e-3:
                out["_note"] = f"mismatch in analyze_kv_transfer[{key}]: got {got_res[key]}, want {ref_res[key]}"
                return out

    out["transfer_rel_err"] = max_trans_err
    out["prefill_rel_err"] = max_pref_err
    return out
