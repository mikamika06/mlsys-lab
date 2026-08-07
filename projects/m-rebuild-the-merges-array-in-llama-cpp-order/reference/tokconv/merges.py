def rebuild_merges(merges, vocab):
    vocab_rank = {token: i for i, token in enumerate(vocab)}
    def get_rank(pair):
        p1, p2 = pair.split(" ")
        t1 = vocab_rank.get(p1, float('inf'))
        t2 = vocab_rank.get(p2, float('inf'))
        return (max(t1, t2), min(t1, t2))
    return sorted(merges, key=get_rank)
