BASE = 1 << 30


def add_limbs(a: list[int], b: list[int]) -> list[int]:
    out = []
    carry = 0

    for i in range(max(len(a), len(b))):
        av = a[i] if i < len(a) else 0
        bv = b[i] if i < len(b) else 0
        total = av + bv + carry
        out.append(total & (BASE - 1))
        carry = total >> 30

    if carry:
        out.append(carry)

    while len(out) > 1 and out[-1] == 0:
        out.pop()

    return out
