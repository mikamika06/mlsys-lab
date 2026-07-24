import random

def _reference(blocks):
    labels = []
    for block in blocks:
        pre = bool(block.get("pre_norm", False))
        post = bool(block.get("post_norm", False))
        if not (pre or post):
            raise ValueError("Block must have at least one norm flag set")
        if pre and not post:
            labels.append("pre")
        elif post and not pre:
            labels.append("post")
        else:  # both True
            labels.append("sandwich")
    return labels

def grade(sol, fx) -> dict:
    ok = 1.0
    for _ in range(20):
        n_blocks = random.randint(1, 10)
        blocks = []
        for _ in range(n_blocks):
            # Ensure at least one flag is True
            pre = random.choice([True, False])
            post = random.choice([True, False]) if pre else random.choice([True, False])
            if not (pre or post):
                post = True
            blocks.append({"pre_norm": pre, "post_norm": post})
        try:
            got = sol.classify_norm_wiring(blocks)
            ref = _reference(blocks)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
