import os
import json
import numpy as np


def get_ref_matmul_shape():
    return (64, 64, 64)


def get_ref_best_latency(workdir=None):
    return 0.005


def get_ref_default_latency(workdir=None):
    return 0.020


def get_ref_top_candidates(workdir=None):
    return [
        {"id": 0, "latency": 0.005},
        {"id": 1, "latency": 0.007},
        {"id": 2, "latency": 0.009},
        {"id": 3, "latency": 0.012},
        {"id": 4, "latency": 0.015},
    ]
