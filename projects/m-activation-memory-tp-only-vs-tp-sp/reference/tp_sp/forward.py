import numpy as np

def all_gather(sharded_tensors):
    out = np.concatenate(sharded_tensors, axis=0)
    return [out.copy() for _ in sharded_tensors]

def reduce_scatter(tensors):
    summed = np.sum(tensors, axis=0)
    tp_size = len(tensors)
    return list(np.split(summed, tp_size, axis=0))

def all_reduce(tensors):
    summed = np.sum(tensors, axis=0)
    return [summed.copy() for _ in tensors]

def forward_tp(X_list, W1_list, W2_list):
    Y1 = [x @ w1 for x, w1 in zip(X_list, W1_list)]
    Y2 = [y1 @ w2 for y1, w2 in zip(Y1, W2_list)]
    return all_reduce(Y2)

def forward_sp(X_sharded_list, W1_list, W2_list):
    X_gathered = all_gather(X_sharded_list)
    Y1 = [x @ w1 for x, w1 in zip(X_gathered, W1_list)]
    Y2 = [y1 @ w2 for y1, w2 in zip(Y1, W2_list)]
    return reduce_scatter(Y2)
