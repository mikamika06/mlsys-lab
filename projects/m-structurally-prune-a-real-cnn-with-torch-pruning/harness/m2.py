import torch
import ref


def check(workdir):
    from cnnprune.prune import SimpleCNN, structural_prune
    out = {"size_ratio": 1.0}
    try:
        net = SimpleCNN()
        pruned = structural_prune(net, 0.5)
        orig_size = sum(p.numel() for p in net.parameters())
        pruned_size = sum(p.numel() for p in pruned.parameters())
        ratio = float(pruned_size) / float(orig_size)
        out["size_ratio"] = ratio
        if ratio <= 0.6:
            out["size_ratio"] = float(ratio)
        else:
            out["_note"] = f"size ratio {ratio} is too high (> 0.6)"
    except Exception as e:
        out["_note"] = f"error: {str(e)[:100]}"
    return out
