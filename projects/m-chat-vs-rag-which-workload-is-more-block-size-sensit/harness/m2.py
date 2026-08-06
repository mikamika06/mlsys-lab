import math
import ref


def check(workdir):
    from block_sensitivity.lookup import model_block_table_lookup_cost, simulate_decode_step_latency

    out = {"lookup_cost_rel_err": 1.0, "decode_latency_rel_err": 1.0}

    seq_len = 4096
    block_size = 16
    num_layers = 32
    num_heads = 32
    head_dim = 128
    bw = 900.0

    want_lookup = ref.oracle_lookup(seq_len, block_size, num_layers)
    want_decode = ref.oracle_decode(seq_len, block_size, num_layers, num_heads, head_dim, bw)

    try:
        got_lookup = model_block_table_lookup_cost(seq_len, block_size, num_layers)
        g_cyc = float(got_lookup.get("estimated_cycles", 0))
        w_cyc = float(want_lookup["estimated_cycles"])
        err_lookup = abs(g_cyc - w_cyc) / max(w_cyc, 1e-9)
        out["lookup_cost_rel_err"] = err_lookup
        if err_lookup > 0.01:
            out["_note_lookup"] = f"got cycles {g_cyc}, want {w_cyc}"
    except Exception as e:
        out["_note_lookup_err"] = f"model_block_table_lookup_cost raised {e}"

    try:
        got_decode = simulate_decode_step_latency(seq_len, block_size, num_layers, num_heads, head_dim, bw)
        g_lat = float(got_decode.get("total_decode_time_us", 0.0))
        w_lat = float(want_decode["total_decode_time_us"])
        err_decode = abs(g_lat - w_lat) / max(w_lat, 1e-9)
        out["decode_latency_rel_err"] = err_decode
        if err_decode > 0.01:
            out["_note_decode"] = f"got decode time {g_lat}, want {w_lat}"
    except Exception as e:
        out["_note_decode_err"] = f"simulate_decode_step_latency raised {e}"

    return out
