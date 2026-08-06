from typing import Any, Dict, List
from seqcomm.formulas import (
    ulysses_comm_volume_per_layer,
    ring_comm_volume_per_layer,
    usp_comm_volume_per_layer,
)


def sweep_comm_costs(
    seq_lens: List[int],
    hidden_dims: List[int],
    world_sizes: List[int],
    head_counts: List[int],
    dtype_bytes: int = 2,
) -> List[Dict[str, Any]]:
    """Sweep communication volumes across Ulysses, Ring, and USP configurations."""
    results = []
    for N in seq_lens:
        for D in hidden_dims:
            for P in world_sizes:
                for H in head_counts:
                    ulysses_valid = (H % P == 0)
                    u_vol = ulysses_comm_volume_per_layer(N, D, P, dtype_bytes) if ulysses_valid else None
                    r_vol = ring_comm_volume_per_layer(N, D, P, dtype_bytes)

                    usp_configs = []
                    for u_deg in [d for d in range(1, P + 1) if P % d == 0]:
                        r_deg = P // u_deg
                        if H % u_deg == 0:
                            vol = usp_comm_volume_per_layer(N, D, P, u_deg, r_deg, dtype_bytes)
                            usp_configs.append({
                                "ulysses_degree": u_deg,
                                "ring_degree": r_deg,
                                "volume_bytes": vol,
                            })

                    results.append({
                        "seq_len": N,
                        "hidden_dim": D,
                        "world_size": P,
                        "num_heads": H,
                        "ulysses_valid": ulysses_valid,
                        "ulysses_volume_bytes": u_vol,
                        "ring_volume_bytes": r_vol,
                        "usp_configs": usp_configs,
                    })
    return results
