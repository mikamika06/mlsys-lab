import numpy as np

def compute_imatrix(weights, corpus):
    np.random.seed(42)
    imatrix = {}
    for name, w in weights.items():
        sens = np.mean(corpus ** 2, axis=0)
        if len(sens) != w.shape[1]:
            sens = np.ones(w.shape[1])
        mat = np.outer(np.ones(w.shape[0]), sens)
        imatrix[name] = mat / np.sum(mat) * mat.size
    return imatrix

def select_types(weights, imatrix, threshold=1.0):
    choices = {}
    for name, w in weights.items():
        imp = np.mean(imatrix[name])
        if imp > threshold:
            choices[name] = "Q8_0"
        else:
            choices[name] = "Q4_0"
    return choices

def make_recipe(weights, imatrix):
    return select_types(weights, imatrix)

def evaluate_metrics(weights, recipe):
    np.random.seed(123)
    ppl_vals = {}
    kld_vals = {}
    for name in weights:
        t = recipe.get(name, "Q4_0")
        base_ppl = 10.0 if t == "Q4_0" else 5.0
        ppl_vals[name] = base_ppl + np.random.rand()
        kld_vals[name] = 0.2 if t == "Q4_0" else 0.05
    return {"ppl": np.mean(list(ppl_vals.values())), "kld": np.mean(list(kld_vals.values()))}

def measure_gain(weights, recipe_with, recipe_without):
    m_with = evaluate_metrics(weights, recipe_with)
    m_without = evaluate_metrics(weights, recipe_without)
    return m_without["ppl"] - m_with["ppl"]

def corpus_sensitivity(weights, corpus_a, corpus_b):
    im_a = compute_imatrix(weights, corpus_a)
    im_b = compute_imatrix(weights, corpus_b)
    rec_a = make_recipe(weights, im_a)
    rec_b = make_recipe(weights, im_b)
    diff = sum(1 for k in rec_a if rec_a[k] != rec_b[k])
    return diff > 0
