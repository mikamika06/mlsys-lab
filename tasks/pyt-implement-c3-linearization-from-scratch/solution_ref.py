def c3(graph, cls_index, names):
    def bases(i):
        return [j for j in range(len(names)) if graph[i][j] != 0]

    def lin(i):
        direct = bases(i)
        if not direct:
            return [i]
        seqs = [lin(b) for b in direct]
        seqs.append(list(direct))
        return [i] + merge(seqs)

    def merge(seqs):
        seqs = [list(s) for s in seqs if s]
        out = []
        while seqs:
            candidate = None
            for seq in seqs:
                head = seq[0]
                if all(head not in other[1:] for other in seqs):
                    candidate = head
                    break
            if candidate is None:
                raise TypeError("inconsistent hierarchy")
            out.append(candidate)
            new = []
            for seq in seqs:
                if seq and seq[0] == candidate:
                    seq = seq[1:]
                if seq:
                    new.append(seq)
            seqs = new
        return out

    return [names[i] for i in lin(cls_index)]
