import numpy as np

def collect_imatrix(weights, corpus):
    imatrix = {}
    for name, w in weights.items():
        sens = np.mean(corpus ** 2, axis=0)
        if len(sens) != w.shape[1]:
            sens = np.ones(w.shape[1])
        mat = np.outer(np.ones(w.shape[0]), sens)
        imatrix[name] = mat / (np.sum(mat) + 1e-8) * mat.size
    return imatrix
