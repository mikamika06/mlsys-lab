import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    from roofline.memory import compute_decode_bytes_per_step
    from roofline.predictor import predict_decode_throughput

    out = {"memory_matched": 0.0, "roofline_matched": 0.0}

    mem_ok = True
    for cfg in ref.CONFIGS:
        for batch_size in [1, 16]:
            for ctx_len in [1024, 4096]:
                want = ref.ref_compute_decode_bytes_per_step(cfg, batch_size, ctx_len)
                try:
                    got = compute_decode_bytes_per_step(cfg, batch_size, ctx_len)
                except Exception as e:
                    out["_note"] = f"memory computation error: {type(e).__name__}: {str(e)}"
                    return out

                rel_err = abs(got - want) / want if want != 0 else 0.0
                if rel_err >= 1e-5:
                    mem_ok = False
                    out["_note"] = f"memory mismatch: got {got}, want {want}"
                    break

    if mem_ok:
        out["memory_matched"] = 1.0

    roofline_ok = True
    for cfg in ref.CONFIGS:
        for batch_size in [1, 64]:
            want_pred = ref.ref_predict_decode_throughput(cfg, batch_size, 2048, 312.0, 2000.0)
            try:
                got_pred = predict_decode_throughput(cfg, batch_size, 2048, 312.0, 2000.0)
            except Exception as e:
                out["_note"] = f"predictor error: {type(e).__name__}: {str(e)}"
                return out

            for k in ["step_latency_sec", "tokens_per_sec", "operational_intensity"]:
                w_val = want_pred[k]
                g_val = got_pred.get(k, 0.0)
                rel_err = abs(g_val - w_val) / w_val if w_val != 0 else 0.0
                if rel_err >= 1e-5:
                    roofline_ok = False
                    out["_note"] = f"predictor key {k} mismatch: got {g_val}, want {w_val}"
                    break
            if got_pred.get("bound") != want_pred["bound"]:
                roofline_ok = False
                out["_note"] = f"bound mismatch: got {got_pred.get('bound')}, want {want_pred['bound']}"
                break

    if roofline_ok:
        out["roofline_matched"] = 1.0

    return out
