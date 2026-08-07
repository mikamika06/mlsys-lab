def get_version_matrix():
    return {
        "sm_80": {"cuda": "11.8", "torch": "2.1.2", "python": "3.10"},
        "sm_89": {"cuda": "12.1", "torch": "2.2.0", "python": "3.10"},
        "sm_90": {"cuda": "12.3", "torch": "2.3.0", "python": "3.11"}
    }

def analyze_gates():
    return {
        "blocked_by": ["compiler_mismatch", "unsupported_sm"],
        "resolved": True
    }
