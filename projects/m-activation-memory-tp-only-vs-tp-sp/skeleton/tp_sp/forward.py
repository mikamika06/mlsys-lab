import numpy as np

def all_gather(sharded_tensors):
    raise NotImplementedError

def reduce_scatter(tensors):
    raise NotImplementedError

def all_reduce(tensors):
    raise NotImplementedError

def forward_tp(X_list, W1_list, W2_list):
    raise NotImplementedError

def forward_sp(X_sharded_list, W1_list, W2_list):
    raise NotImplementedError
