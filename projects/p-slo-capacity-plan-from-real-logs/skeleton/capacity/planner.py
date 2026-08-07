import json
import numpy as np


def parse_metrics_series(metrics_jsonl_path):
    raise NotImplementedError


def parse_bench_results(bench_json_path):
    raise NotImplementedError


def calculate_goodput(bench_data, max_p95_latency):
    raise NotImplementedError


def find_knee_capacity(bench_data, target_p95):
    raise NotImplementedError


def compute_required_replicas(target_rps, single_replica_capacity, headroom_factor=1.2):
    raise NotImplementedError


def compute_cost_per_million_tokens(replica_count, hourly_cost_per_replica, rps, avg_output_tokens):
    raise NotImplementedError


def compute_prefix_cache_impact(target_rps, base_single_replica_capacity, hit_rate, speedup_factor, headroom_factor=1.2):
    raise NotImplementedError
