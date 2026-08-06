import ref
import numpy as np

def check(workdir):
    from treespec import builder
    
    trees = ref.generate_trees(50)
    matched_bytes = 0
    total_bytes = 0
    rng = np.random.RandomState(1337)
    
    for parents in trees:
        root_pos = int(rng.randint(0, 100))
        w_mask, w_pos = ref.build_tree_mask_and_positions(parents, root_pos)
        
        try:
            g_mask, g_pos = builder.build_tree_mask_and_positions(list(parents), root_pos)
            if w_mask.shape == g_mask.shape and w_mask.dtype == g_mask.dtype:
                if np.array_equal(w_mask, g_mask):
                    matched_bytes += w_mask.nbytes
            if w_pos.shape == g_pos.shape and w_pos.dtype == g_pos.dtype:
                if np.array_equal(w_pos, g_pos):
                    matched_bytes += w_pos.nbytes
        except Exception:
            pass
            
        total_bytes += w_mask.nbytes + w_pos.nbytes
        
    return {"byte_exact_fraction": float(matched_bytes) / total_bytes if total_bytes > 0 else 0.0}
