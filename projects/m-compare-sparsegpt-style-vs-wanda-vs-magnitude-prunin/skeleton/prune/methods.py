def magnitude_prune(w, sparsity):
    raise NotImplementedError


def wanda_prune(w, X, sparsity):
    raise NotImplementedError


def sparsegpt_prune(w, X, sparsity):
    raise NotImplementedError


def evaluate_quality(w_orig, w_pruned, X):
    raise NotImplementedError


def compare_methods(w, X, sparsity):
    raise NotImplementedError
