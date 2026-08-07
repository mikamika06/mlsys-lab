import ref

def evaluate_modules(module_sets):
    return ref.simulate_modules(module_sets)

def analyze_alpha_scaling(alphas, ranks):
    return ref.simulate_alpha_scaling(alphas, ranks)

def find_pareto_front(results):
    return ref.simulate_pareto(results)

def verify_second_domain(config):
    return ref.simulate_second_domain(config)
