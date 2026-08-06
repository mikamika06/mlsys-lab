import numpy as np


def merge_and_unload(state_dict):
    new_dict = {}
    lora_prods = {}
    for k in list(state_dict.keys()):
        if ".lora_A" in k:
            base_k = k.replace(".lora_A", ".weight")
            b_k = k.replace(".lora_A", ".lora_B")
            if b_k in state_dict:
                lora_prods[base_k] = np.matmul(state_dict[b_k], state_dict[k])

    for k, v in state_dict.items():
        if "lora" in k or "adapter" in k:
            continue
        if k in lora_prods:
            new_dict[k] = v + lora_prods[k]
        else:
            new_dict[k] = v
    return new_dict
