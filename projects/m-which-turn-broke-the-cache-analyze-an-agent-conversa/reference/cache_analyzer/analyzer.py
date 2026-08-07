def find_cache_break_turn(trace):
    turns = trace["turns"]
    prev_tokens = []
    for t_idx, turn in enumerate(turns):
        tokens = turn["prompt_text"].split()
        if t_idx > 0:
            common = 0
            min_len = min(len(prev_tokens), len(tokens))
            for a, b in zip(prev_tokens[:min_len], tokens[:min_len]):
                if a == b:
                    common += 1
                else:
                    break
            if common < len(prev_tokens):
                return t_idx
        prev_tokens = tokens
    return -1
