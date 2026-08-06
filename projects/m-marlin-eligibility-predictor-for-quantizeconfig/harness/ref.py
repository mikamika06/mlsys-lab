CONFIGS = [
    {"bits": 4, "group_size": 128, "sym": True, "desc_act": False, "expected_eligible": True},
    {"bits": 8, "group_size": -1, "sym": True, "desc_act": False, "expected_eligible": True},
    {"bits": 4, "group_size": 128, "sym": True, "desc_act": True, "expected_eligible": False},
    {"bits": 4, "group_size": 16, "sym": True, "desc_act": False, "expected_eligible": False},
    {"bits": 3, "group_size": 128, "sym": True, "desc_act": False, "expected_eligible": False},
    {"bits": 4, "group_size": 128, "sym": False, "desc_act": False, "expected_eligible": False},
]
