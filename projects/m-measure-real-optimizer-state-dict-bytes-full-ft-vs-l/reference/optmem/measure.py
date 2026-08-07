import torch


def get_optimizer_state_bytes(optimizer):
    total = 0
    for state_dict in optimizer.state.values():
        for v in state_dict.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def compare_full_vs_lora(full_model, lora_model):
    def run_step(model):
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable)
        for p in trainable:
            p.grad = torch.ones_like(p)
        opt.step()
        return get_optimizer_state_bytes(opt)

    f_bytes = run_step(full_model)
    l_bytes = run_step(lora_model)

    return {
        "full_bytes": f_bytes,
        "lora_bytes": l_bytes,
        "size_ratio": l_bytes / f_bytes if f_bytes else 0.0
    }
