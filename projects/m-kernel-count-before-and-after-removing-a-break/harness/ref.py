import re

CODE_WITH_BREAK = """
from ctypes import c_void_p, c_long
import torch
from torch import empty_strided

@pointwise(size_hints=[1024])
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    pass

@pointwise(size_hints=[1024])
def triton_poi_fused_mul_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    pass
"""

CODE_FUSED = """
from ctypes import c_void_p, c_long
import torch

@pointwise(size_hints=[1024])
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    pass
"""

def parse_kernels(code: str) -> list[dict]:
    kernels = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def triton_") or line.startswith("def cpp_"):
            m = re.match(r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):", line)
            if m:
                name = m.group(1)
                args_raw = m.group(2).split(",")
                params = []
                for a in args_raw:
                    clean = a.split(":")[0].strip()
                    if clean and not clean.isupper() and "numel" not in clean:
                        params.append(clean)
                kernels.append({"name": name, "params": params})
    return kernels

def count_kernels(code: str) -> int:
    return len(parse_kernels(code))

def calc_bytes(ops: list[dict], element_size: int, num_elements: int, fused: bool) -> int:
    if not fused:
        return sum(len(op["inputs"]) + 1 for op in ops) * element_size * num_elements
    created = set(op["output"] for op in ops)
    used = set(inp for op in ops for inp in op["inputs"])
    return (len(used - created) + len(created - used)) * element_size * num_elements
