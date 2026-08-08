import numpy as np

def generate_scenarios():
    np.random.seed(42)
    scenarios = []
    for i in range(3):
        arrival_rates = np.linspace(10, 100, 20) + i * 5
        scenarios.append({"arrival_rates": arrival_rates, "capacity": 50.0})
    return scenarios

SCENARIOS = generate_scenarios()

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

def compute_metrics(latencies, preemptions):
    p99 = np.percentile(latencies, 99)
    total_pree = np.sum(preemptions)
    return float(p99), float(total_pree)

def correlate(latencies, preemptions):
    if len(latencies) < 2:
        return 0.0
    corr = np.corrcoef(latencies, preemptions)[0, 1]
    return float(corr)
