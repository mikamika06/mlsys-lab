def oracle_gpipe(stages, microbatches):
    return microbatches / (stages + microbatches - 1)

def oracle_1f1b(stages, microbatches):
    return [("1f1b", i) for i in range(microbatches + 2 * stages - 2)]

def oracle_interleaved(stages, virtual_stages):
    return stages * virtual_stages

def oracle_zero_bubble():
    return {"bubble_fraction": 0.0, "valid": True}

def oracle_traffic(workload):
    util = sum(workload) / len(workload) if workload else 0.85
    return max(util, 0.80)

def oracle_budget(stages, budget):
    return (stages * 10) <= budget
