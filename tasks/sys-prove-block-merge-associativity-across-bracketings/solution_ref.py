import math


def _merge(a, b):
    m1, s1 = a
    m2, s2 = b
    if m1 > m2:
        m = m1
    else:
        m = m2
    return (
        m,
        s1 * math.exp(m1 - m) + s2 * math.exp(m2 - m),
    )


def _left(parts):
    out = parts[0]
    for part in parts[1:]:
        out = _merge(out, part)
    return out


def _right(parts):
    if len(parts) == 1:
        return parts[0]
    return _merge(parts[0], _right(parts[1:]))


def _check(parts):
    vals = [_left(parts), _right(parts)]
    if len(parts) >= 3:
        if len(parts) > 3:
            m_sub = _merge(_merge(parts[0], parts[1]), _merge(parts[2], parts[3]))
        else:
            m_sub = _merge(_merge(parts[0], parts[1]), parts[2])
        vals.append(m_sub)
    a = vals[0]
    all_ok = True
    for b in vals[1:]:
        diff0 = a[0] - b[0]
        if diff0 < 0:
            diff0 = -diff0
        diff1 = a[1] - b[1]
        if diff1 < 0:
            diff1 = -diff1
        if diff0 > 1e-6 or diff1 > 1e-6:
            all_ok = False
            break
    return all_ok


def check_block_merge_associativity(rows):
    answer = []
    for row in rows:
        valid = True
        n = len(row)
        for count in range(2, 6):
            cuts = [int(i * n / count) for i in range(count + 1)]
            blocks = []
            for i in range(count):
                block = row[cuts[i]:cuts[i + 1]]
                m = block[0]
                for val in block[1:]:
                    if val > m:
                        m = val
                s = 0.0
                for val in block:
                    s += math.exp(val - m)
                blocks.append((float(m), float(s)))
            if not _check(blocks):
                valid = False
                break
        answer.append(valid)
    return answer
