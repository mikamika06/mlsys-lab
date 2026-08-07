import sys

def rank_memory_spaces():
    reg = 0
    sh = [0] * 10
    l2 = [0.0] * 10
    glob = {i: 0 for i in range(10)}
    sizes = {
        'register': sys.getsizeof(reg),
        'shared': sys.getsizeof(sh),
        'L2': sys.getsizeof(l2),
        'global': sys.getsizeof(glob)
    }
    return sorted(sizes, key=sizes.get)
