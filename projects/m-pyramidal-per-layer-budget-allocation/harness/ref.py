import numpy as np
from pyrkv.allocation import compute_pyramidal_allocation
from pyrkv.bakeoff import run_bakeoff
from pyrkv.curve import compute_accuracy_curve

NUM_LAYERS = 16
TOTAL_BUDGET = 4800
MIN_BUDGET = 150
PROMPTS = ["needle in haystack one", "needle in haystack two", "needle in haystack three"]
STRATEGIES = [{"name": "uniform"}, {"name": "pyramidal"}, {"name": "hybrid"}]
RATIOS = [0.25, 0.5, 0.75, 1.0]

def get_reference_allocation():
    return compute_pyramidal_allocation(NUM_LAYERS, TOTAL_BUDGET, MIN_BUDGET)

def get_reference_bakeoff():
    return run_bakeoff(PROMPTS, TOTAL_BUDGET, STRATEGIES)

def dummy_eval(r):
    return float(r) * 0.95

def get_reference_curve():
    return compute_accuracy_curve(RATIOS, dummy_eval)
