import math


def _flatten(obj):
    """Recursively flattens nested lists/tuples or array-like objects."""
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten(item)
    elif hasattr(obj, "tolist"):
        yield from _flatten(obj.tolist())
    else:
        yield obj


def _compute_expected(layers, prune_ratio):
    """Computes ground truth layer sparsities and most pruned layer index."""
    sizes = []
    abs_pool_list = []
    for w in layers:
        flat = list(_flatten(w))
        sizes.append(len(flat))
        for val in flat:
            abs_pool_list.append(abs(float(val)))

    N = len(abs_pool_list)
    if N == 0:
        return [], 0

    k = int(round(prune_ratio * N))

    indexed_pool = [(abs_pool_list[i], i) for i in range(N)]
    sorted_indexed = sorted(indexed_pool, key=lambda x: x[0])
    order = [item[1] for item in sorted_indexed]

    pruned = [False] * N
    for i in range(k):
        pruned[order[i]] = True

    cum_sizes = [0]
    curr = 0
    for s in sizes:
        curr += s
        cum_sizes.append(curr)
    offsets = cum_sizes

    sparsity_list = []
    for i in range(len(layers)):
        start = offsets[i]
        end = offsets[i + 1]
        sub = pruned[start:end]
        total_true = sum(1 for val in sub if val)
        count = len(sub)
        mean_val = float(total_true) / count if count > 0 else 0.0
        sparsity_list.append(mean_val)

    most_pruned = 0
    if len(sparsity_list) > 0:
        max_val = sparsity_list[0]
        for i in range(1, len(sparsity_list)):
            if sparsity_list[i] > max_val:
                max_val = sparsity_list[i]
                most_pruned = i

    return sparsity_list, most_pruned


def _to_plain_lists(layers):
    """Converts layers (which might be nested lists or numpy arrays) to plain Python nested lists."""
    def _convert(obj):
        if hasattr(obj, "tolist"):
            return _convert(obj.tolist())
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        return float(obj)

    if isinstance(layers, (list, tuple)):
        return [_convert(l) for l in layers]
    return layers


def _get_sol_results(sol, layers_list, prune_ratio):
    """Invokes global_threshold_layer_sparsity on sol and extracts output values."""
    sol_sparsity = None
    sol_most_pruned = None

    fn_names = [
        "global_threshold_layer_sparsity",
        "global_threshold_sparsity",
        "layer_sparsity",
    ]

    for name in fn_names:
        if hasattr(sol, name):
            try:
                res = getattr(sol, name)(layers_list, prune_ratio)
                if isinstance(res, dict):
                    if "sparsity" in res:
                        sol_sparsity = res["sparsity"]
                    if "most_pruned_layer" in res:
                        sol_most_pruned = res["most_pruned_layer"]
                    break
                elif isinstance(res, (tuple, list)) and len(res) >= 2:
                    sol_sparsity = res[0]
                    sol_most_pruned = res[1]
                    break
            except Exception:
                pass

    if hasattr(sol_sparsity, "tolist"):
        sol_sparsity = sol_sparsity.tolist()
    if isinstance(sol_sparsity, (list, tuple)):
        sol_sparsity = [float(x) for x in sol_sparsity]

    if sol_most_pruned is not None:
        try:
            sol_most_pruned = int(sol_most_pruned)
        except Exception:
            sol_most_pruned = None

    return sol_sparsity, sol_most_pruned


def grade(sol, fx) -> dict:
    test_cases = []

    # Extract fixture layers from fx if provided
    fx_layers = None
    if fx is not None:
        if isinstance(fx, dict):
            if all(f"layer{i}" in fx for i in range(4)):
                fx_layers = [fx[f"layer{i}"] for i in range(4)]
            elif "weights" in fx:
                fx_layers = fx["weights"]
            elif len(fx) > 0:
                fx_layers = [fx[k] for k in sorted(fx.keys())]
        elif isinstance(fx, (list, tuple)) and len(fx) > 0:
            fx_layers = list(fx)

    if fx_layers is not None:
        test_cases.append((fx_layers, 0.5))

    # Synthetic test cases
    test_cases.extend([
        (
            [
                [0.01, 0.01, 0.01, 0.01],
                [2.0, 2.0, 2.0, 2.0],
            ],
            0.5,
        ),
        (
            [
                [[0.01, 0.02], [0.03, 0.04]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            0.5,
        ),
        (
            [
                [0.0, 0.1, 0.0, 0.0, 0.2],
                [[1.0, 2.0], [3.0, 4.0]],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            0.3,
        ),
        (
            [
                [0.01],
                [1.0],
                [0.05],
            ],
            0.66,
        ),
    ])

    sparsity_all_passed = True
    most_pruned_all_passed = True

    for layers, ratio in test_cases:
        layers_plain = _to_plain_lists(layers)
        exp_sparsity, exp_most_pruned = _compute_expected(layers_plain, ratio)
        sol_sparsity, sol_most_pruned = _get_sol_results(sol, layers_plain, ratio)

        if sol_sparsity is None or len(sol_sparsity) != len(exp_sparsity):
            sparsity_all_passed = False
        else:
            for s_sol, s_exp in zip(sol_sparsity, exp_sparsity):
                if not math.isclose(s_sol, s_exp, abs_tol=1e-5):
                    sparsity_all_passed = False
                    break

        if sol_most_pruned is None or sol_most_pruned != exp_most_pruned:
            most_pruned_all_passed = False

    return {
        "sparsity_exact": 1 if sparsity_all_passed else 0,
        "most_pruned_layer_exact": 1 if most_pruned_all_passed else 0,
    }
