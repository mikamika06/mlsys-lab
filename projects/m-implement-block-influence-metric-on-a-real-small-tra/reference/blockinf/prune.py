def select_layers_to_remove(bi_scores, num_remove):
    indexed = sorted(enumerate(bi_scores), key=lambda x: x[1])
    remove_indices = [idx for idx, _ in indexed[:num_remove]]
    return sorted(remove_indices)
