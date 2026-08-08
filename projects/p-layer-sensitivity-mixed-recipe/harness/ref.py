import numpy as np


def get_fixture():
    np.random.seed(42)
    model = {
        "l1": np.random.randn(32, 32).astype(np.float32),
        "l2": np.random.randn(32, 64).astype(np.float32) * 0.1,
        "l3": np.random.randn(64, 16).astype(np.float32),
        "l4": np.random.randn(16, 10).astype(np.float32) * 5.0,
    }
    x = np.random.randn(16, 32).astype(np.float32)
    return model, x


def dict_close(d1, d2):
    if d1.keys() != d2.keys(): return False
    for k in d1:
        if d1[k].keys() != d2[k].keys(): return False
        for p in d1[k]:
            if abs(d1[k][p] - d2[k][p]) > 1e-4: return False
    return True


def quantize_tensor(tensor, bits):
    if bits >= 32:
        return tensor.copy()
    qmin = 0
    qmax = (1 << bits) - 1
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val == min_val:
        return tensor.copy()
    scale = (max_val - min_val) / qmax
    q = np.round((tensor - min_val) / scale)
    q = np.clip(q, qmin, qmax)
    return q * scale + min_val


def get_size_bytes(shape, bits):
    return int(np.prod(shape) * bits // 8)


def forward(model, x):
    out = x
    for name in sorted(model.keys()):
        out = out @ model[name]
    return out


def measure_sensitivity(model, x, precisions):
    base_out = forward(model, x)
    sens = {name: {} for name in model}
    for name in model:
        for p in precisions:
            if p >= 32:
                sens[name][p] = 0.0
                continue
            q_model = {k: v.copy() for k, v in model.items()}
            q_model[name] = quantize_tensor(model[name], p)
            q_out = forward(q_model, x)
            sens[name][p] = float(np.mean((base_out - q_out) ** 2))
    return sens


def build_recipe(model_shapes, sens, budget, precisions):
    sorted_p = sorted(precisions, reverse=True)
    recipe = {name: sorted_p[0] for name in model_shapes}

    def current_size():
        return sum(get_size_bytes(model_shapes[n], recipe[n]) for n in model_shapes)

    while current_size() > budget:
        best_downgrade = None
        best_score = float('inf')

        for name in sorted(model_shapes.keys()):
            curr_p = recipe[name]
            curr_idx = sorted_p.index(curr_p)
            if curr_idx == len(sorted_p) - 1:
                continue
            next_p = sorted_p[curr_idx + 1]

            size_saved = get_size_bytes(model_shapes[name], curr_p) - get_size_bytes(model_shapes[name], next_p)
            if size_saved <= 0:
                continue

            err_increase = sens[name][next_p] - sens[name][curr_p]
            score = err_increase / size_saved

            if score < best_score:
                best_score = score
                best_downgrade = (name, next_p)

        if best_downgrade is None:
            break
        recipe[best_downgrade[0]] = best_downgrade[1]

    return recipe


def evaluate_recipe(model, x, recipe):
    q_model = {}
    for name, w in model.items():
        q_model[name] = quantize_tensor(w, recipe[name])
    base_out = forward(model, x)
    q_out = forward(q_model, x)
    return float(np.mean((base_out - q_out) ** 2))


def compare_recipes(model, x, budget, precisions):
    shapes = {k: v.shape for k, v in model.items()}
    sorted_p = sorted(precisions, reverse=True)

    best_uni_p = sorted_p[-1]
    for p in sorted_p:
        total_size = sum(get_size_bytes(shapes[n], p) for n in shapes)
        if total_size <= budget:
            best_uni_p = p
            break
    uni_recipe = {n: best_uni_p for n in shapes}

    sens = measure_sensitivity(model, x, precisions)
    mix_recipe = build_recipe(shapes, sens, budget, precisions)

    uni_mse = evaluate_recipe(model, x, uni_recipe)
    mix_mse = evaluate_recipe(model, x, mix_recipe)

    return {
        "uniform_mse": uni_mse,
        "mixed_mse": mix_mse,
        "uniform_recipe": uni_recipe,
        "mixed_recipe": mix_recipe
    }
