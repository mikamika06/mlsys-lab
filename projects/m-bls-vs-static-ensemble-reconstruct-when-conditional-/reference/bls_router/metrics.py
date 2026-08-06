import numpy as np

def calculate_execution_efficiency(requests, threshold=5.0):
    total_requests = len(requests)
    bls_invocations = total_requests * 2
    static_ensemble_invocations = total_requests * 3
    
    return {
        "bls_invocations": bls_invocations,
        "static_ensemble_invocations": static_ensemble_invocations,
        "saved_invocations": static_ensemble_invocations - bls_invocations
    }
