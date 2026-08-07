import numpy as np


def reshard_state_dict(state_dicts_source, source_world_size, target_world_size):
    flat_data = []
    for sd in state_dicts_source:
        for k in sorted(sd.keys()):
            flat_data.append(sd[k].flatten())
    if not flat_data:
        return [{} for _ in range(target_world_size)]

    cat_data = np.concatenate(flat_data)
    total_elements = len(cat_data)

    target_dicts = []
    curr_idx = 0
    for r in range(target_world_size):
        base_chunk = total_elements // target_world_size
        remainder = total_elements % target_world_size
        elements_for_rank = base_chunk + (1 if r < remainder else 0)
        chunk = cat_data[curr_idx:curr_idx + elements_for_rank]
        curr_idx += elements_for_rank
        target_dicts.append({"chunk": chunk})
    return target_dicts
