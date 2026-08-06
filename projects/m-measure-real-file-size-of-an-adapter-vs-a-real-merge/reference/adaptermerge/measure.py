import os
import numpy as np


def measure_file_sizes(base_dict, adapter_dict, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    b_path = os.path.join(tmp_dir, "base.npz")
    a_path = os.path.join(tmp_dir, "adapter.npz")
    np.savez(b_path, **base_dict)
    np.savez(a_path, **adapter_dict)
    b_size = os.path.getsize(b_path)
    a_size = os.path.getsize(a_path)

    merged = {}
    for k, v in base_dict.items():
        if k in adapter_dict:
            merged[k] = v + adapter_dict[k]
        elif k.replace(".weight", ".lora_B") in adapter_dict and k.replace(".weight", ".lora_A") in adapter_dict:
            b_key = k.replace(".weight", ".lora_B")
            a_key = k.replace(".weight", ".lora_A")
            merged[k] = v + np.matmul(adapter_dict[b_key], adapter_dict[a_key])
        else:
            merged[k] = v
    m_path = os.path.join(tmp_dir, "merged.npz")
    np.savez(m_path, **merged)
    m_size = os.path.getsize(m_path)
    return {"base_size": b_size, "adapter_size": a_size, "merged_size": m_size, "size_ratio": m_size / (b_size + a_size)}
