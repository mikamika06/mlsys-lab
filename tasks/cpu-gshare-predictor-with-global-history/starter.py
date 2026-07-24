def simulate_branch(pc_list, outcome_list, k):
    """Simulate a gshare predictor with k-bit GHR and PHT of size 2**k.

    pc_list: list of program counter values (int)
    outcome_list: list of actual branch outcomes (1=taken, 0=not taken)
    k: history length in bits (PHT has 2**k entries)

    Returns: number of mispredictions.
    """
    raise NotImplementedError('your code here')
