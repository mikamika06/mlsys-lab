from finetune.memory import compute_training_memory

def classify_hardware_budgets(param_count, budgets):
    supported = []
    for name, capacity_bytes in budgets.items():
        for method in ["full", "lora", "qlora"]:
            req = compute_training_memory(param_count, method)
            if req <= capacity_bytes:
                if method not in supported:
                    supported.append(method)
    return sorted(supported)
