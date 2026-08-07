import numpy as np


def generate_report(results):
    return f"Report: size_ratio={results.get('size_ratio', 0):.2f}, acc_drop={results.get('acc_drop', 0):.4f}"
