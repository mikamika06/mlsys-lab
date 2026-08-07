import numpy as np

def get_tensor_sizes(model):
    return {k: int(w.nbytes) for k, w in model.items()}

def pareto_report(model_variants):
    report = []
    for name, model, acc in model_variants:
        total_size = sum(w.nbytes for w in model.values())
        report.append({"name": name, "size": total_size, "accuracy": acc})
    return report
