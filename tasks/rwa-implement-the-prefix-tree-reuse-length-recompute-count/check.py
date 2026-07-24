import random, string

def _oracle(ops):
    stored = set()
    reuse = []
    recompute = []
    for op, word in ops:
        max_lcp = 0
        for w in stored:
            lcp = 0
            for a, b in zip(word, w):
                if a == b:
                    lcp += 1
                else:
                    break
            if lcp > max_lcp:
                max_lcp = lcp
        reuse.append(max_lcp)
        recompute.append(len(word) - max_lcp)
        if op == 'add':
            stored.add(word)
    return reuse, recompute

def grade(sol, fx) -> dict:
    random.seed(0)
    ok = 1.0
    for _ in range(5):
        ops = []
        n = random.randint(3, 10)
        for i in range(n):
            op = random.choice(['add', 'query'])
            length = random.randint(1, 8)
            word = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
            ops.append((op, word))
        # ensure at least two adds
        if sum(1 for o, _ in ops if o == 'add') < 2:
            ops[0] = ('add', ops[0][1])
            ops[1] = ('add', ops[1][1])
        try:
            got_reuse, got_rec = sol.process_ops(list(ops))
            exp_reuse, exp_rec = _oracle(ops)
            if list(got_reuse) != exp_reuse or list(got_rec) != exp_rec:
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
