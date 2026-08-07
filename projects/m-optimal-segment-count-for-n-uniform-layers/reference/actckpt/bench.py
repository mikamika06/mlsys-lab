from typing import Dict, Any
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential


def measure_checkpoint_execution(
    model_layers: nn.ModuleList,
    x: torch.Tensor,
    num_segments: int
) -> Dict[str, Any]:
    """Measures step time and peak memory allocated during checkpointed execution."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    if num_segments <= 1:
        out = x
        for layer in model_layers:
            out = layer(out)
    else:
        out = checkpoint_sequential(model_layers, num_segments, x, use_reentrant=False)

    loss = out.sum()
    loss.backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        peak_bytes = 0

    elapsed = time.perf_counter() - start_time
    return {"elapsed_sec": float(elapsed), "peak_bytes": int(peak_bytes)}
