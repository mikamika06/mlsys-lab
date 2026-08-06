"""Workload block-size sensitivity and hit-rate regression models."""

from typing import Dict, List, Any


def compare_workload_sensitivity(
    chat_prompt_lengths: List[int],
    rag_prompts: List[List[int]],
    block_sizes: List[int]
) -> Dict[int, Dict[str, Any]]:
    """Analyze block-size sensitivity and fragmentation across Chat and RAG workloads."""
    raise NotImplementedError


def explain_hit_rate_regression(
    shared_prefix_len: int,
    request_suffix_lengths: List[int],
    block_size_1: int,
    block_size_2: int
) -> Dict[str, Any]:
    """Explain why prefix cache hit rate drops when block size doubles."""
    raise NotImplementedError
