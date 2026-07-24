def c3_merge_trace(bases):
    sequences = [list(base.mro()) for base in bases]
    sequences.append(list(bases))
    result = []

    while True:
        sequences = [seq for seq in sequences if seq]
        if not sequences:
            break

        selected = None
        for seq in sequences:
            head = seq[0]
            if all(head not in other[1:] for other in sequences):
                selected = head
                break

        if selected is None:
            raise TypeError("inconsistent hierarchy")

        result.append(selected.__name__)
        for seq in sequences:
            if seq and seq[0] is selected:
                seq.pop(0)

    return result
