"""Deterministic hardware models.

Nothing here touches real hardware. Each model is exact and reproducible, so a
task can gate on a *measured* number — transactions, bank-conflict waves, cache
misses — and get the same answer on every machine.

  gpu.py     software GPU: 32-lane warps, 128-byte coalescing transactions,
             32 shared-memory banks, warp divergence, barriers, warp shuffles
  cuda_c.py  a restricted CUDA-C front end that executes a real `.cu` kernel
             thread-by-thread on that GPU
  cache.py   set-associative LRU cache model over a byte-address trace
  abi.py     pinned LP64 struct layout (size / alignment / offsets)
  simt.py    the plain-python SIMT helper used by analysis-only tasks
"""

from . import abi, cache, simt
from .cuda_c import CudaParseError, CudaProgram
from .gpu import GPU, BANKS, SEGMENT, WARP, ShflRequest, Thread

__all__ = ["GPU", "Thread", "ShflRequest", "WARP", "SEGMENT", "BANKS",
           "CudaProgram", "CudaParseError", "cache", "abi", "simt"]
