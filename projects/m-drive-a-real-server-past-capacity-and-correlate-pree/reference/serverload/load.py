import numpy as np

def simulate_load(scenario):
    rates = scenario["arrival_rates"]
    capacity = scenario["capacity"]
    preemptions = []
    latencies = []
    for r in rates:
        overload = max(0.0, r - capacity)
        pree = float(overload * 2.5 + np.sin(r) * 2.0)
        lat = float(50.0 + overload * 15.0 + np.cos(r) * 5.0)
        preemptions.append(max(0.0, pree))
        latencies.append(max(10.0, lat))
    return {"preemptions": np.array(preemptions), "latencies": np.array(latencies)}
