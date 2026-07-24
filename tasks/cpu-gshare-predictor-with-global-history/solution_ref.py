def simulate_branch(pc_list, outcome_list, k):
    size = 1 << k
    pht = [2] * size
    ghr = 0
    mispredictions = 0
    for pc, outcome in zip(pc_list, outcome_list):
        index = (pc ^ ghr) & (size - 1)
        prediction = 1 if pht[index] >= 2 else 0
        if prediction != outcome:
            mispredictions += 1
        if outcome:
            pht[index] = min(pht[index] + 1, 3)
        else:
            pht[index] = max(pht[index] - 1, 0)
        ghr = ((ghr << 1) | outcome) & (size - 1)
    return mispredictions
