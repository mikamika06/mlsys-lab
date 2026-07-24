import sys


def str_footprint(lengths, max_codepoints):
    result = []
    for n, cp in zip(lengths, max_codepoints):
        if n == 0:
            s = ""
        else:
            s = chr(cp) * n
        result.append(sys.getsizeof(s))
    return result
