import numpy as np

VOCAB_SIZE = 50
NUM_RUNS = 10

def generate_fixtures():
    np.random.seed(123)
    unmasked_drafts = []
    target_probs_list = []
    grammar_masks = []
    for _ in range(NUM_RUNS):
        k = 4
        draft_tokens = list(np.random.randint(0, VOCAB_SIZE, size=k))
        unmasked_drafts.append(draft_tokens)

        tp = []
        for _ in range(k):
            p = np.random.rand(VOCAB_SIZE)
            p /= p.sum()
            tp.append(p)
        target_probs_list.append(tp)

        gmask = []
        for _ in range(k):
            mask = {i: (np.random.rand() > 0.3) for i in range(VOCAB_SIZE)}
            gmask.append(mask)
        grammar_masks.append(gmask)

    return unmasked_drafts, target_probs_list, grammar_masks
