import numpy as np
from decode_prof.model import simulate_decode_metrics
from decode_prof.analysis import find_crossover_batch_size, diagnose_occupancy_limiter

BATCH_SIZES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=int)
HIDDEN_DIM = 4096
WEIGHT_BYTES = 1024 * 1024 * 1024
PEAK_BW = 1500.0 * 1024 * 1024 * 1024
PEAK_FLOP_RATE = 312.0 * 1024 * 1024 * 1024 * 1024

TPUTS, BW_REAL = simulate_decode_metrics(BATCH_SIZES, HIDDEN_DIM, WEIGHT_BYTES, PEAK_BW, PEAK_FLOP_RATE)
CROSSOVER = find_crossover_batch_size(BATCH_SIZES, TPUTS, BW_REAL, PEAK_BW)
OCCUPANCY_METRICS = diagnose_occupancy_limiter(BATCH_SIZES, BW_REAL, PEAK_BW)
