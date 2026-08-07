def cost_model(b, gamma):
    tt = 10.0 + 1.0 * b
    td = 3.0 + 0.5 * b
    tv = 10.0 + 1.2 * b + 0.5 * b * gamma
    return td, tt, tv

def generate_trace():
    return [
        {"drafted": 4, "accepted": 2},
        {"drafted": 4, "accepted": 4},
        {"drafted": 4, "accepted": 0},
        {"drafted": 4, "accepted": 1},
    ]

def generate_requests():
    reqs = []
    for i in range(100):
        domain = "chat" if i % 2 == 0 else "code"
        p_true = 0.8 if domain == "chat" else 0.2
        b = (i % 16) + 1
        reqs.append({"id": i, "b": b, "p_true": p_true, "domain": domain})
    return reqs
