def diamond_merge(cls):
    def merge(seqs):
        seqs = [list(s) for s in seqs if s]
        result = []
        while seqs:
            chosen = None
            for seq in seqs:
                head = seq[0]
                if not any(head in other[1:] for other in seqs):
                    chosen = head
                    break
            if chosen is None:
                raise TypeError("inconsistent hierarchy")
            result.append(chosen)
            new_seqs = []
            for seq in seqs:
                if seq and seq[0] is chosen:
                    seq = seq[1:]
                if seq:
                    new_seqs.append(seq)
            seqs = new_seqs
        return result

    def linearize(c):
        if not c.__bases__:
            return [c]
        parents = [linearize(base) for base in c.__bases__]
        parents.append(list(c.__bases__))
        return [c] + merge(parents)

    return [c.__name__ for c in linearize(cls)]
