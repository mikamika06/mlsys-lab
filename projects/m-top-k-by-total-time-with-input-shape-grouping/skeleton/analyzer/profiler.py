import json


def load_trace(path):
    raise NotImplementedError


def aggregate_by_shape(events):
    raise NotImplementedError


def top_k_by_total_time(events, k=5):
    raise NotImplementedError
