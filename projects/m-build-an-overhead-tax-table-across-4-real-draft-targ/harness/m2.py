import ref


def check(workdir):
    from tax.reports import build_overhead_tax_table, find_optimal_gamma

    out = {
        "table_matched": 0.0,
        "optimal_matched": 0.0,
        "rel_err": 1.0,
    }

    gammas = [2, 3, 5]
    want_table = ref.build_overhead_tax_table(ref.PAIRS_DATA, gammas)
    got_table = build_overhead_tax_table(ref.PAIRS_DATA, gammas)

    max_err = 0.0
    table_ok = True

    if len(got_table) != len(want_table):
        table_ok = False
        out["_note"] = f"table length mismatch: got {len(got_table)}, want {len(want_table)}"
    else:
        for w_row, g_row in zip(want_table, got_table):
            if w_row["pair_id"] != g_row.get("pair_id") or w_row["gamma"] != g_row.get("gamma"):
                table_ok = False
            for k in ["effective_latency_ms", "speedup", "overhead_tax"]:
                w_val = float(w_row[k])
                g_val = float(g_row.get(k, 0.0))
                err = abs(g_val - w_val) / max(1.0, abs(w_val))
                max_err = max(max_err, err)
                if err > 1e-5:
                    table_ok = False

    optimal_ok = True
    for pair in ref.PAIRS_DATA:
        want_opt = ref.find_optimal_gamma(pair, 6)
        got_opt = find_optimal_gamma(pair, 6)
        if got_opt is None or got_opt.get("gamma") != want_opt["gamma"]:
            optimal_ok = False
        for k in ["overhead_tax", "speedup"]:
            w_val = float(want_opt[k])
            g_val = float(got_opt.get(k, 0.0) if got_opt else 0.0)
            err = abs(g_val - w_val) / max(1.0, abs(w_val))
            max_err = max(max_err, err)
            if err > 1e-5:
                optimal_ok = False

    out["table_matched"] = 1.0 if table_ok else 0.0
    out["optimal_matched"] = 1.0 if optimal_ok else 0.0
    out["rel_err"] = float(max_err)
    return out
