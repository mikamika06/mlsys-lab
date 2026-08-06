import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from longctx_eval.diagnostics import diagnose_models
    except ImportError:
        return {"rel_err": 1.0, "modes_match": 0.0, "_note": "could not import module"}

    accs, ents, lengths, acc_thresh, ent_thresh = ref.get_m2_data()
    want = ref.diagnose_models(accs, ents, lengths, acc_thresh, ent_thresh)

    try:
        got = diagnose_models(accs, ents, lengths, acc_thresh, ent_thresh)
    except Exception as e:
        return {"rel_err": 1.0, "modes_match": 0.0, "_note": f"crashed: {e}"}

    if not isinstance(got, list) or len(got) != len(want):
        return {"rel_err": 1.0, "modes_match": 0.0, "_note": "list length mismatch"}

    max_err = 0.0
    modes_ok = 1.0

    for w, g in zip(want, got):
        if w['mode'] != g.get('mode'):
            modes_ok = 0.0
        ws, gs = w['slope'], g.get('slope', 0.0)
        err = abs(ws - gs) / (abs(ws) + 1e-9)
        if err > max_err:
            max_err = err

    return {"rel_err": float(max_err), "modes_match": modes_ok}
