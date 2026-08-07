from radixtree.tree import TokenRadixTree

def simulate_cache(traces, cache_type="radix", block_size=16):
    if cache_type == "radix":
        tree = TokenRadixTree()
        total_hits = 0
        total_tokens = 0
        for trace in traces:
            total_tokens += len(trace)
            matched, _ = tree.longest_match(trace)
            total_hits += len(matched)
            tree.insert(trace)
        return {"hits": total_hits, "total": total_tokens, "hit_rate": total_hits / max(1, total_tokens)}
    else:
        seen_blocks = set()
        total_hits = 0
        total_tokens = 0
        for trace in traces:
            total_tokens += len(trace)
            hits_in_trace = 0
            for i in range(0, len(trace), block_size):
                block = tuple(trace[i:i+block_size])
                if block in seen_blocks:
                    hits_in_trace += len(block)
                else:
                    seen_blocks.add(block)
            total_hits += hits_in_trace
        return {"hits": total_hits, "total": total_tokens, "hit_rate": total_hits / max(1, total_tokens)}
