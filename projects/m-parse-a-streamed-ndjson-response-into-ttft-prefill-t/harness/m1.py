import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from streammetrics.parser import parse_stream_metrics

    dataset = ref.generate_dataset(num_traces=10, seed=123)
    errs = []

    for item in dataset:
        want = ref.ref_parse_stream_metrics(item["stream"])
        try:
            got = parse_stream_metrics(item["stream"])
        except Exception as e:
            return {"rel_err": 1.0, "_note": f"Exception raised during parse: {e}"}

        for k in ["ttft", "prefill_tok_per_sec", "decode_tok_per_sec"]:
            w_val = want[k]
            g_val = got.get(k, 0.0)
            if abs(w_val) > 1e-9:
                errs.append(abs(g_val - w_val) / abs(w_val))
            else:
                errs.append(abs(g_val - w_val))

    max_err = max(errs) if errs else 0.0
    return {"rel_err": max_err}
