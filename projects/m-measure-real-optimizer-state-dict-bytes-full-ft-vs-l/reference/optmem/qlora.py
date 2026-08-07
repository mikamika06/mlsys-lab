import torch
from optmem.measure import get_optimizer_state_bytes


def measure_8bit_delta(model, optimizer_cls_32, optimizer_cls_8):
    trainable = [p for p in model.parameters() if p.requires_grad]

    opt32 = optimizer_cls_32(trainable)
    for p in trainable:
        p.grad = torch.ones_like(p)
    opt32.step()
    bytes_32 = get_optimizer_state_bytes(opt32)

    opt8 = optimizer_cls_8(trainable)
    for p in trainable:
        p.grad = torch.ones_like(p)
    opt8.step()
    bytes_8 = get_optimizer_state_bytes(opt8)

    return {
        "bytes_32": bytes_32,
        "bytes_8": bytes_8,
        "delta_bytes": bytes_32 - bytes_8
    }


def verify_qlora_optimizer_clean(model, optimizer):
    frozen_ids = {id(p) for p in model.parameters() if not p.requires_grad}
    for group in optimizer.param_groups:
        for p in group['params']:
            if id(p) in frozen_ids:
                if p in optimizer.state and len(optimizer.state[p]) > 0:
                    return False
    return True
