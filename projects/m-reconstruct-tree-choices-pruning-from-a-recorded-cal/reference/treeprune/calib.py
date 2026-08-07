import numpy as np

def reconstruct_tree_choices(fixture):
    choices = []
    for row in fixture:
        sorted_idx = np.argsort(row)[::-1]
        choices.append(sorted_idx[:3].tolist())
    return choices
