import numpy as np

def build_tree_mask_and_positions(parents, root_pos):
    """
    Builds the tree attention mask and position IDs.
    `parents` is a list where parents[i] is the parent index of node i.
    Node 0 is the root, so parents[0] = -1.
    Returns (attention_mask, position_ids) as numpy arrays of type np.int32.
    """
    raise NotImplementedError


def select_longest_path(parents, accepted_nodes):
    """
    Selects the longest continuous path of accepted nodes starting from the root.
    `accepted_nodes` is a boolean array of length N.
    Returns a list of node indices in the path.
    """
    raise NotImplementedError
