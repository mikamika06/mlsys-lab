import ref

def check(workdir):
    from app.components import MockModel, MockFSM
    from app.decoder import generate_safe
    model = MockModel(42)

    fsm1 = MockFSM()
    t1 = generate_safe(model, fsm1, 20)
    m = {"normal_ok": 1.0 if len(t1) == 9 and fsm1.state == 9 else 0.0}

    fsm2 = MockFSM()
    t2 = generate_safe(model, fsm2, 5)
    m["truncation_handled"] = 1.0 if len(t2) == 9 and fsm2.state == 9 else 0.0

    fsm3 = MockFSM()
    t3 = generate_safe(model, fsm3, 9)
    m["exact_budget_ok"] = 1.0 if len(t3) == 9 and fsm3.state == 9 else 0.0
    return m
