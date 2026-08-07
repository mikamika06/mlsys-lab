import numpy as np


class AdaptivePolicy:
    def __init__(self, model, tracker, min_speedup=1.02):
        self.model = model
        self.tracker = tracker
        self.min_speedup = float(min_speedup)

    def decide(self, domain, batch_size, max_gamma=8):
        tau = self.tracker.get_acceptance_rate(domain)
        best_g = self.model.optimal_gamma(tau, max_gamma=max_gamma, batch_size=batch_size)
        if best_g > 0:
            sp = self.model.expected_speedup(best_g, tau, batch_size)
            if sp >= self.min_speedup:
                return best_g, True
        return 0, False

    def simulate_request(self, domain, batch_size, gamma, accepted_count, base_step_time=10.0):
        if gamma == 0:
            time_taken = base_step_time * (1.0 + 0.05 * (batch_size - 1))
            tokens_generated = 1
        else:
            step_cost_factor = self.model.expected_step_cost(gamma, batch_size) / self.model.target_step_cost
            time_taken = base_step_time * step_cost_factor
            tokens_generated = accepted_count + 1
            self.tracker.record(domain, accepted_count, gamma)
        return time_taken, tokens_generated

    def evaluate_p95_and_throughput(self, traffic_stream):
        latencies = []
        total_tokens = 0
        total_time = 0.0

        for req in traffic_stream:
            domain = req["domain"]
            batch_size = req["batch_size"]
            max_gamma = req.get("max_gamma", 8)
            sim_accepted = req["sim_accepted"]
            base_step_time = req.get("base_step_time", 10.0)

            gamma, active = self.decide(domain, batch_size, max_gamma=max_gamma)
            if not active:
                g_exec = 0
                acc_exec = 0
            else:
                g_exec = gamma
                acc_exec = min(sim_accepted, g_exec)

            t_step, tok_step = self.simulate_request(domain, batch_size, g_exec, acc_exec, base_step_time)
            token_lat = t_step / float(tok_step)
            latencies.append(token_lat)
            total_tokens += tok_step
            total_time += t_step

        p95_latency = float(np.percentile(latencies, 95))
        throughput = float(total_tokens) / float(total_time) if total_time > 0 else 0.0

        return {
            "p95_latency": p95_latency,
            "throughput": throughput,
            "requests_processed": len(traffic_stream)
        }
