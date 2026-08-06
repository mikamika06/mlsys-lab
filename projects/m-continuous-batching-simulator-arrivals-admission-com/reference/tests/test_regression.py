from cbsim.simulator import Request, admission_predicate

def test_admission_predicate_limits():
    r1 = Request(1, 0, 100, 10)
    r2 = Request(2, 0, 100, 10)
    active = [r1]
    assert admission_predicate(active, r2, max_batch_size=4, max_capacity=150) is False

def test_admission_predicate_batch_size():
    r1 = Request(1, 0, 10, 10)
    r2 = Request(2, 0, 10, 10)
    active = [r1]
    assert admission_predicate(active, r2, max_batch_size=1, max_capacity=1000) is False
