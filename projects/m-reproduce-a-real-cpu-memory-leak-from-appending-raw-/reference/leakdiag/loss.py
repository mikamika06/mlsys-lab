import sys
import torch


def measure_loss_memory(steps):
    leaky_list = []
    clean_list = []
    for i in range(steps):
        t = torch.tensor([float(i)], requires_grad=True)
        leaky_list.append(t)
        clean_list.append(t.item())
    size_leaky = sys.getsizeof(leaky_list) + sum(sys.getsizeof(x) for x in leaky_list)
    size_clean = sys.getsizeof(clean_list) + sum(sys.getsizeof(x) for x in clean_list)
    return {"size_leaky": float(size_leaky), "size_clean": float(size_clean), "ratio": float(size_leaky / max(1, size_clean))}
