"""Block table lookup cost model and decode step latency breakdown."""

from typing import Dict, Any


def model_block_table_lookup_cost(
    seq_len: int,
    block_size: int,
    num_layers: int,
    table_levels: int = 2
) -> Dict[str, Any]:
    """Model block table lookup operations and cycle overhead per decode step."""
    num_blocks = (seq_len + block_size - 1) // block_size
    lookups_per_layer = num_blocks * table_levels
    total_lookups = num_layers * lookups_per_layer

    cycles_per_lookup = 4
    total_cycles = total_lookups * cycles_per_lookup

    return {
        "num_blocks": num_blocks,
        "lookups_per_layer": lookups_per_layer,
        "total_lookups": total_lookups,
        "estimated_cycles": total_cycles
    }


def simulate_decode_step_latency(
    seq_len: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    mem_bandwidth_gbps: float,
    clock_ghz: float = 1.5
) -> Dict[str, Any]:
    """Simulate total decode step time comparing KV memory transfer and block lookup cost."""
    lookup_info = model_block_table_lookup_cost(seq_len, block_size, num_layers)

    bytes_per_elem = 2
    kv_bytes_per_token = 2 * num_layers * num_heads * head_dim * bytes_per_elem
    total_kv_bytes = seq_len * kv_bytes_per_token

    mem_time_us = (total_kv_bytes / (mem_bandwidth_gbps * 1e9)) * 1e6
    lookup_time_us = (lookup_info["estimated_cycles"] / (clock_ghz * 1e9)) * 1e6

    total_decode_time_us = mem_time_us + lookup_time_us

    return {
        "kv_bytes_transferred": total_kv_bytes,
        "mem_time_us": mem_time_us,
        "lookup_time_us": lookup_time_us,
        "total_decode_time_us": total_decode_time_us,
        "lookup_overhead_pct": (lookup_time_us / total_decode_time_us) * 100.0 if total_decode_time_us > 0 else 0.0
    }
