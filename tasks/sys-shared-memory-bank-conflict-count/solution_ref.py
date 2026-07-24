def bank_conflict_degree(accesses):
    result = []
    for warp in accesses:
        banks = [0] * 32
        for address in warp:
            banks[address % 32] += 1
        result.append(max(banks))
    return result
