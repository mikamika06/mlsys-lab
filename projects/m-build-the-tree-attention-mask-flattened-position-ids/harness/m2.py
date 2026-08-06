import ref
import numpy as np

def check(workdir):
    from treespec import builder, analysis
    
    out = {"paths_correct": 0.0, "sweep_correct": 0.0}
    trees = ref.generate_trees(30)
    path_ok = 0
    rng = np.random.RandomState(888)
    
    for parents in trees:
        n = len(parents)
        accepted = rng.choice([True, False], size=n, p=[0.7, 0.3])
        accepted[0] = True
        
        want = ref.select_longest_path(parents, accepted)
        try:
            got = builder.select_longest_path(list(parents), accepted.copy())
            if list(got) == list(want):
                path_ok += 1
        except Exception:
            pass
            
    out["paths_correct"] = float(path_ok) / len(trees)
    
    records = []
    for _ in range(100):
        records.append({
            "tree_width": int(rng.choice([2, 4, 8])),
            "step": int(rng.randint(1, 100)),
            "accepted_length": float(rng.randint(1, 5))
        })
        
    sums = {}
    counts = {}
    for r in records:
        w = r["tree_width"]
        sums[w] = sums.get(w, 0.0) + r["accepted_length"]
        counts[w] = counts.get(w, 0) + 1
    want_sweep = {w: sums[w] / counts[w] for w in sums}
    
    try:
        got_sweep = analysis.analyze_sweep(records)
        if got_sweep == want_sweep:
            out["sweep_correct"] = 1.0
    except Exception:
        pass
        
    return out
