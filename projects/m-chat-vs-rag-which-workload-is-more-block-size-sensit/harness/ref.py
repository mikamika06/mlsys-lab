"""Reference fixture generation and oracle calculations."""

import random
from typing import Dict, List, Any
from block_sensitivity.workload import compare_workload_sensitivity, explain_hit_rate_regression
from block_sensitivity.lookup import model_block_table_lookup_cost, simulate_decode_step_latency


def get_chat_prompts(seed: int = 42) -> List[int]:
    rng = random.Random(seed)
    return [rng.randint(50, 300) for _ in range(25)]


def get_rag_prompts(seed: int = 42) -> List[List[int]]:
    rng = random.Random(seed)
    prefix_len = 1050
    prefix = [rng.randint(1, 1000) for _ in range(prefix_len)]
    prompts = []
    for _ in range(10):
        suffix_len = rng.randint(40, 200)
        suffix = [rng.randint(1001, 2000) for _ in range(suffix_len)]
        prompts.append(prefix + suffix)
    return prompts


def oracle_sensitivity(chat_lengths: List[int], rag_prompts: List[List[int]], block_sizes: List[int]) -> Dict[int, Dict[str, Any]]:
    return compare_workload_sensitivity(chat_lengths, rag_prompts, block_sizes)


def oracle_explain(prefix_len: int, suffixes: List[int], bs1: int, bs2: int) -> Dict[str, Any]:
    return explain_hit_rate_regression(prefix_len, suffixes, bs1, bs2)


def oracle_lookup(seq_len: int, block_size: int, num_layers: int) -> Dict[str, Any]:
    return model_block_table_lookup_cost(seq_len, block_size, num_layers)


def oracle_decode(seq_len: int, block_size: int, num_layers: int, num_heads: int, head_dim: int, bw: float) -> Dict[str, Any]:
    return simulate_decode_step_latency(seq_len, block_size, num_layers, num_heads, head_dim, bw)
