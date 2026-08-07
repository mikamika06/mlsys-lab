import numpy as np


def compute_rank_file_size(tensor_shapes, world_size, rank, dtype_bytes=4):
    total_elements = sum(int(np.prod(shape)) for shape in tensor_shapes)
    base_chunk = total_elements // world_size
    remainder = total_elements % world_size
    elements_for_rank = base_chunk + (1 if rank < remainder else 0)
    return elements_for_rank * dtype_bytes
