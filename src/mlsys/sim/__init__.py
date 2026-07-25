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
# AST node types are part of the surface too: a few tasks ask the learner to
# reason about the parsed program, not just run it.
from .cuda_c import (Block, CudaParseError, CudaProgram, For, If, SharedDecl,
                     VarDecl, While)
from .gpu import (GPU, BANKS, CYC_ALU, CYC_MEM, CYC_SMEM, SEGMENT, WARP,
                  ShflRequest, Thread, occupancy, latency_hiding,
                  SM_MAX_WARPS, SM_REGS, SM_SMEM)

__all__ = ["GPU", "Thread", "ShflRequest", "WARP", "SEGMENT", "BANKS",
           "CYC_MEM", "CYC_SMEM", "CYC_ALU",
           "CudaProgram", "CudaParseError", "Block", "For", "If", "While",
           "VarDecl", "SharedDecl",
           "cache", "abi", "simt",
           "occupancy", "latency_hiding", "SM_REGS", "SM_SMEM", "SM_MAX_WARPS"]
