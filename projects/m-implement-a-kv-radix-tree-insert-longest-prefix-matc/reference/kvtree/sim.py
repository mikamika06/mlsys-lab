from kvtree.tree import RadixTree

def compare_hit_rates(traces, capacity):
    tree = RadixTree()
    hits = 0
    total = 0
    for trace in traces:
        for prompt in trace:
            total += len(prompt)
            _, matched_len = tree.match_prefix(prompt)
            hits += matched_len
            rem = prompt[matched_len:]
            if rem:
                tree.insert(rem, list(range(len(rem))))
    return hits / max(total, 1)
