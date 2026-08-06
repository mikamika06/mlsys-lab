"""Analyzer module for identifying imbalanced branches."""

def identify_imbalanced_branch(kernel_data):
    branches = kernel_data["branches"]
    return max(branches, key=lambda b: b["divergence_score"])["branch_id"]
