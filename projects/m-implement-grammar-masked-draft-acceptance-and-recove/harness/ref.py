import numpy as np

np.random.seed(42)

def make_batch(batch_size=10, seq_len=4, vocab_size=16):
    draft_tokens = []
    target_probs = []
    draft_probs = []
    grammar_masks = []
    random_samples = []

    for _ in range(batch_size):
        dt = np.random.randint(0, vocab_size, size=seq_len).tolist()
        tp = []
        dp = []
        gm = []
        rs = np.random.rand(seq_len).tolist()
        for i in range(seq_len):
            p = np.random.dirichlet(np.ones(vocab_size))
            q = np.random.dirichlet(np.ones(vocab_size))
            m = np.random.rand(vocab_size) > 0.3
            tp.append(p)
            dp.append(q)
            gm.append(m)
        draft_tokens.append(dt)
        target_probs.append(tp)
        draft_probs.append(dp)
        grammar_masks.append(gm)
        random_samples.append(rs)

    return draft_tokens, target_probs, draft_probs, grammar_masks, random_samples


BATCH_DRAFT_TOKENS, BATCH_TARGET_PROBS, BATCH_DRAFT_PROBS, BATCH_GRAMMAR_MASKS, BATCH_RANDOM_SAMPLES = make_batch()


def compute_accepted_length(draft_tokens, target_probs, draft_probs, grammar_masks, random_samples):
    accepted_length = 0
    for i, token in enumerate(draft_tokens):
        if not grammar_masks[i][token]:
            break
        p = target_probs[i][token]
        q = draft_probs[i][token]
        if p >= q or random_samples[i] < (p / q):
            accepted_length += 1
        else:
            break
    return accepted_length


def measure_acceptance_loss(batch_draft_tokens, batch_target_probs, batch_draft_probs, batch_grammar_masks, batch_random_samples):
    total_loss = 0.0
    n = len(batch_draft_tokens)
    if n == 0:
        return 0.0

    for i in range(n):
        len_masked = compute_accepted_length(
            batch_draft_tokens[i],
            batch_target_probs[i],
            batch_draft_probs[i],
            batch_grammar_masks[i],
            batch_random_samples[i]
        )
        unmasked = [np.ones_like(m, dtype=bool) for m in batch_grammar_masks[i]]
        len_unmasked = compute_accepted_length(
            batch_draft_tokens[i],
            batch_target_probs[i],
            batch_draft_probs[i],
            unmasked,
            batch_random_samples[i]
        )
        total_loss += (len_unmasked - len_masked)

    return float(total_loss / n)
