import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    import ref
    from kvcache.config import LayerConfig, ModelConfig
    from kvcache.cache import HybridKVCache
    from kvcache.engine import HybridAttentionEngine

    m = {"memory_saving_above_threshold": 0.0, "output_equivalence": 0.0}

    try:
        l_configs = []
        for i in range(8):
            is_sliding = i not in (0, 7)
            ws = 16 if is_sliding else None
            l_configs.append(
                LayerConfig(layer_id=i, is_sliding=is_sliding, window_size=ws)
            )
        cfg = ModelConfig(
            num_layers=8, num_heads=4, head_dim=32, layer_configs=l_configs
        )

        q, k, v = ref.generate_synthetic_data(128, cfg.num_heads, cfg.head_dim)

        hyb_cache = HybridKVCache(cfg)
        hyb_engine = HybridAttentionEngine(cfg, hyb_cache)

        ref_cache = ref.FullKVCache(cfg.num_layers)
        ref_engine = ref.FullAttentionEngine(cfg, ref_cache)

        match_all = True
        for l_idx in range(8):
            hyb_out = hyb_engine.process_sequence(q, k, v, l_idx)
            ref_out = ref_engine.process_sequence(q, k, v, l_idx)
            if not np.allclose(hyb_out, ref_out, atol=1e-4, rtol=1e-4):
                match_all = False
                break

        if match_all:
            m["output_equivalence"] = 1.0

        ratio = hyb_cache.memory_saving_ratio()
        m["memory_saving_above_threshold"] = float(ratio)
    except Exception:
        pass

    return m
