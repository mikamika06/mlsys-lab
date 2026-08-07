KERNELS = [
    "x = tl.load(ptr0)\ny = x + 1",
    "a = tl.load(ptr1)\nb = a * 2\nc = tl.maximum(b, 0)",
    "m = dot(x, y)\nn = m + z",
]


def parse_kernel(code: str):
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    ops = []
    for line in lines:
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            ops.append({"lhs": lhs.strip(), "rhs": rhs.strip()})
    return ops


def reconstruct_chain(code: str):
    ops = parse_kernel(code)
    chain = []
    for op in ops:
        rhs = op["rhs"]
        if "tl.load" in rhs:
            chain.append("aten.load")
        elif "tl.fadd" in rhs or "+" in rhs:
            chain.append("aten.add")
        elif "tl.fmul" in rhs or "*" in rhs:
            chain.append("aten.mul")
        elif "tl.maximum" in rhs:
            chain.append("aten.relu")
        elif "dot" in rhs:
            chain.append("aten.mm")
        else:
            chain.append("aten.unknown")
    return chain
