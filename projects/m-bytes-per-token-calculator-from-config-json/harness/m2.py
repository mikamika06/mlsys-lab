import importlib.util
import os
import ref
from kvcalc.ratio import cache_ratio
from kvcalc.concurrency import max_concurrency


def _load_ref(mod_name, rel_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    path = os.path.join(root, "reference", *rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check(workdir):
    out = {"ratio_match": 0.0, "concurrency_match": 0.0}
    cfg_mla = ref.CONFIGS[1]
    cfg_gqa = ref.CONFIGS[0]
    
    ref_ratio_mod = _load_ref("ref_ratio", ["kvcalc", "ratio.py"])
    ref_conc_mod = _load_ref("ref_conc", ["kvcalc", "concurrency.py"])
    
    want_ratio = ref_ratio_mod.cache_ratio(cfg_mla, cfg_gqa, 2)
    try:
        got_ratio = cache_ratio(cfg_mla, cfg_gqa, 2)
        if abs(got_ratio - want_ratio) < 1e-5:
            out["ratio_match"] = 1.0
        else:
            out["_note"] = f"ratio got {got_ratio}, want {want_ratio}"
    except Exception as e:
        out["_note"] = f"ratio raised {type(e).__name__}"
        
    budget = 16 * 1024 * 1024 * 1024
    b_tok = 2048
    seq_len = 4096
    overhead = 1024 * 1024 * 1024
    
    want_conc = ref_conc_mod.max_concurrency(budget, b_tok, seq_len, overhead)
    try:
        got_conc = max_concurrency(budget, b_tok, seq_len, overhead)
        if got_conc == want_conc:
            out["concurrency_match"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"concurrency got {got_conc}, want {want_conc}"
    except Exception as e:
        if "_note" not in out:
            out["_note"] = f"concurrency raised {type(e).__name__}"
            
    return out
