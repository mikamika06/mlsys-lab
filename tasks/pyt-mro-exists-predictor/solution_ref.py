def mro_exists(graph):
    def merge(seqs):
        seqs = [list(seq) for seq in seqs if seq]
        result = []

        while seqs:
            candidate = None
            for seq in seqs:
                head = seq[0]
                if all(head not in other[1:] for other in seqs):
                    candidate = head
                    break

            if candidate is None:
                return None

            result.append(candidate)
            next_seqs = []
            for seq in seqs:
                if seq and seq[0] == candidate:
                    seq = seq[1:]
                if seq:
                    next_seqs.append(seq)
            seqs = next_seqs

        return result

    memo = {}

    def compute(node, active):
        if node in memo:
            return memo[node]
        if node in active:
            return None

        active = set(active)
        active.add(node)

        parent_mros = []
        for base in graph[node]:
            parent_mro = compute(base, active)
            if parent_mro is None:
                return None
            parent_mros.append(parent_mro)

        merged = merge(parent_mros + [list(graph[node])])
        if merged is None:
            return None

        memo[node] = [node] + merged
        return memo[node]

    return compute(0, set()) is not None
