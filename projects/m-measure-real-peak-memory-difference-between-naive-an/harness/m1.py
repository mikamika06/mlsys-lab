import ref

def check(workdir):
    from fusedmem.memory import measure_peak_memory_diff
    ratios = []
    for x in ref.TEST_INPUTS:
        ratio = measure_peak_memory_diff(x)
        ratios.append(ratio)
    avg_ratio = float(sum(ratios) / len(ratios))
    return {"size_ratio": avg_ratio}
