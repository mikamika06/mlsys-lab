import sys


def append_copy_count(n):
    lst = []
    copies = 0
    empty_overhead = sys.getsizeof([]) - [].__sizeof__()
    pointer_size = 8
    previous_capacity = (sys.getsizeof(lst) - empty_overhead) // pointer_size

    for _ in range(n):
        old_len = len(lst)
        lst.append(None)
        current_capacity = (sys.getsizeof(lst) - empty_overhead) // pointer_size
        if current_capacity != previous_capacity:
            copies += old_len
            previous_capacity = current_capacity

    return copies
