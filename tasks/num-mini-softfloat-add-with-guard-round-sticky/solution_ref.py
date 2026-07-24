def _shr_sticky(x, n):
    if n <= 0:
        return x
    if n >= x.bit_length() + 4:
        return 1 if x else 0
    lost = x & ((1 << n) - 1)
    return (x >> n) | (1 if lost else 0)


def fp32_add_bits(a_bits: int, b_bits: int) -> int:
    def unpack(x):
        return (x >> 31) & 1, (x >> 23) & 0xFF, x & 0x7FFFFF

    sa, ea0, fa = unpack(a_bits)
    sb, eb0, fb = unpack(b_bits)

    if ea0 == 255 and fa:
        return 0x7FC00000
    if eb0 == 255 and fb:
        return 0x7FC00000
    if ea0 == 255:
        return a_bits
    if eb0 == 255:
        return b_bits

    if ea0 == 0:
        ea = -126
        ma = fa
    else:
        ea = ea0 - 127
        ma = (1 << 23) | fa

    if eb0 == 0:
        eb = -126
        mb = fb
    else:
        eb = eb0 - 127
        mb = (1 << 23) | fb

    if ma == 0 and mb == 0:
        return 0

    if ea < eb:
        sa, sb = sb, sa
        ea, eb = eb, ea
        ma, mb = mb, ma

    e = ea
    ma <<= 3
    mb <<= 3
    mb = _shr_sticky(mb, ea - eb)

    if sa == sb:
        m = ma + mb
        sign = sa
    else:
        if ma >= mb:
            m = ma - mb
            sign = sa
        else:
            m = mb - ma
            sign = sb

    if m == 0:
        return 0

    while m < (1 << 26):
        m <<= 1
        e -= 1

    if m >= (1 << 27):
        m = _shr_sticky(m, 1)
        e += 1

    if e < -126:
        m = _shr_sticky(m, -126 - e)
        e = -126

    frac_part = m & 7
    mant = m >> 3

    if (frac_part & 4) and ((frac_part & 3) or (mant & 1)):
        mant += 1

    if mant >= (1 << 24):
        mant >>= 1
        e += 1

    if e > 127:
        return (sign << 31) | (255 << 23)

    if e == -126 and mant < (1 << 23):
        exp = 0
        frac = mant
    else:
        exp = e + 127
        frac = mant & 0x7FFFFF

    return (sign << 31) | (exp << 23) | frac
