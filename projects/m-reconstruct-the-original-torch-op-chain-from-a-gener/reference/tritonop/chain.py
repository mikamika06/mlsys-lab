from tritonop.parser import parse_kernel


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
