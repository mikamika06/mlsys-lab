"""Workload block-size sensitivity and hit-rate regression models."""

import math
from typing import Dict, List, Any


def compare_workload_sensitivity(
    chat_prompt_lengths: List[int],
    rag_prompts: List[List[int]],
    block_sizes: List[int]
) -> Dict[int, Dict[str, Any]]:
    """Analyze block-size sensitivity and fragmentation across Chat and RAG workloads."""
    results = {}

    rag_lengths = [len(p) for p in rag_prompts]

    for bs in block_sizes:
        chat_blocks = sum(math.ceil(l / bs) for l in chat_prompt_lengths)
        chat_allocated_tokens = chat_blocks * bs
        chat_used_tokens = sum(chat_prompt_lengths)
        chat_waste = chat_allocated_tokens - chat_used_tokens
        chat_frag_rate = chat_waste / chat_allocated_tokens if chat_allocated_tokens > 0 else 0.0

        rag_blocks = sum(math.ceil(l / bs) for l in rag_lengths)
        rag_allocated_tokens = rag_blocks * bs
        rag_used_tokens = sum(rag_lengths)
        rag_waste = rag_allocated_tokens - rag_used_tokens
        rag_frag_rate = rag_waste / rag_allocated_tokens if rag_allocated_tokens > 0 else 0.0

        if len(rag_prompts) > 1:
            common_prefix_len = 0
            first = rag_prompts[0]
            for idx in range(min(len(p) for p in rag_prompts)):
                if all(p[idx] == first[idx] for p in rag_prompts):
                    common_prefix_len += 1
                else:
                    break
        else:
            common_prefix_len = rag_lengths[0] if rag_lengths else 0

        cached_blocks = common_prefix_len // bs
        unaligned_prefix_loss = common_prefix_len - (cached_blocks * bs)

        results[bs] = {
            "chat_frag_rate": chat_frag_rate,
            "rag_frag_rate": rag_frag_rate,
            "chat_total_blocks": chat_blocks,
            "rag_total_blocks": rag_blocks,
            "unaligned_prefix_loss": unaligned_prefix_loss,
            "more_sensitive": "rag" if rag_frag_rate > chat_frag_rate or unaligned_prefix_loss > 0 else "chat"
        }

    return results


def explain_hit_rate_regression(
    shared_prefix_len: int,
    request_suffix_lengths: List[int],
    block_size_1: int,
    block_size_2: int
) -> Dict[str, Any]:
    """Explain why prefix cache hit rate drops when block size doubles."""
    cached_tokens_bs1 = (shared_prefix_len // block_size_1) * block_size_1
    cached_tokens_bs2 = (shared_prefix_len // block_size_2) * block_size_2

    num_requests = len(request_suffix_lengths)
    total_tokens = sum(shared_prefix_len + s for s in request_suffix_lengths)

    hits_bs1 = num_requests * cached_tokens_bs1
    hits_bs2 = num_requests * cached_tokens_bs2

    hit_rate_bs1 = hits_bs1 / total_tokens if total_tokens > 0 else 0.0
    hit_rate_bs2 = hits_bs2 / total_tokens if total_tokens > 0 else 0.0

    return {
        "shared_prefix_len": shared_prefix_len,
        "bs1": block_size_1,
        "bs2": block_size_2,
        "cached_tokens_bs1": cached_tokens_bs1,
        "cached_tokens_bs2": cached_tokens_bs2,
        "hit_rate_bs1": hit_rate_bs1,
        "hit_rate_bs2": hit_rate_bs2,
        "hit_rate_drop": hit_rate_bs1 - hit_rate_bs2,
        "unaligned_tail_loss": cached_tokens_bs1 - cached_tokens_bs2
    }
