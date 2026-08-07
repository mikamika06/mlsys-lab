import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from kvcache.cache import HybridKVCache
    from kvcache.engine import HybridAttentionEngine

    m = {"attention_outputs_match": 0.0, "logits_match": 0.0}

    try:
        cfg = ref.make_sample_config(num_layers=2, window_size=5)
        q, k, v = ref.generate_synthetic_data(20, cfg.num_heads, cfg.head_dim)

        ref_cache = ref.FullKVCache(cfg.num_layers)
        ref_engine = ref.FullAttentionEngine(cfg, ref_cache)

        hyb_cache = HybridKVCache(cfg)
        hyb_engine = HybridAttentionEngine(cfg, hyb_cache)

        ref_out = ref_engine.process_sequence(q, k, v, layer_idx=1)
        hyb_out = hyb_engine.process_sequence(q, k, v, layer_idx=1)

        if np.allclose(ref_out, hyb_out, atol=1e-5, rtol=1e-5):
            m["attention_outputs_match"] = 1.0

        ref_out_full = ref_engine.process_sequence(q, k, v, layer_idx=0)
        hyb_out_full = hyb_engine.process_sequence(q, k, v, layer_idx=0)

        if np.allclose(ref_out_full, hyb_out_full, atol=1e-5, rtol=1e-5):
            m["logits_match"] = 1.0
    except Exception:
        pass

    return m
