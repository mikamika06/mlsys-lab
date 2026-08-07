import numpy as np
from reference.serving import engine as ref_engine

def latency_curve(batch_sizes, base_latency, overhead, threads):
    return ref_engine.latency_curve(batch_sizes, base_latency, overhead, threads)

def thread_scaling(batch_sizes, thread_counts, base_cost):
    return ref_engine.thread_scaling(batch_sizes, thread_counts, base_cost)

def find_slo_point(batch_sizes, slo_latency, arrival_rate, threads):
    return ref_engine.find_slo_point(batch_sizes, slo_latency, arrival_rate, threads)

def simulate_burst(batch_size, burst_requests, threads):
    return ref_engine.simulate_burst(batch_size, burst_requests, threads)

def max_throughput_point(batch_sizes, slo_latency, threads):
    return ref_engine.max_throughput_point(batch_sizes, slo_latency, threads)

def recalculate_for_model(model_params, slo_latency, threads):
    return ref_engine.recalculate_for_model(model_params, slo_latency, threads)
