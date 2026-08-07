import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from quantrec.crossover import crossover_batch_size


def recommend_scheme(workload):
    batch_size = workload["batch_size"]
    bw = workload["bandwidth_gbps"]
    tflops = workload["tflops_w16"]
    crossover = crossover_batch_size(bw, tflops)
    if batch_size < crossover:
        return "W4A16"
    return "W8A8"
