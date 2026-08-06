from typing import List, Tuple


def build_tinybert_layer_mapping(student_layers: int, teacher_layers: int) -> List[Tuple[int, int]]:
    """Generates TinyBERT mapping from student layer index to teacher layer index."""
    if student_layers <= 0 or teacher_layers <= 0:
        raise ValueError("Layer counts must be positive integers.")
    if teacher_layers % student_layers != 0:
        raise ValueError("Teacher layers must be divisible by student layers for TinyBERT mapping.")

    stride = teacher_layers // student_layers
    mapping = [(s_idx, (s_idx + 1) * stride) for s_idx in range(student_layers)]
    return mapping
