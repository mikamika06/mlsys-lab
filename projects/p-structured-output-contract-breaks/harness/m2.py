import ref

def check(workdir):
    from app.components import MockModel, MockFSM
    from app.decoder import decode_with_schema
    model = MockModel(42)

    fsm1 = MockFSM()
    t1 = decode_with_schema(model, fsm1, 10)
    m = {"generates_valid": 1.0 if len(t1) == 9 and fsm1.state == 9 else 0.0}

    fsm2 = MockFSM()
    t2 = decode_with_schema(model, fsm2, 5)
    m["stops_at_max"] = 1.0 if len(t2) == 5 and fsm2.state == 5 else 0.0
    return m
