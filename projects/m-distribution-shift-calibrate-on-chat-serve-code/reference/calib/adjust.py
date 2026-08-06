import numpy as np


def adjust_scales(chat_scales, shift_matrix):
    return chat_scales * np.clip(shift_matrix, 0.5, 2.0)
