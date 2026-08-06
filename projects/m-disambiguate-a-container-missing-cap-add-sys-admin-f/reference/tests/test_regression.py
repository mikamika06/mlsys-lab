from profilediag.classifier import classify_error

def test_classifier_basic_sys_admin():
    log = "Error: perf_event_open failed with Permission denied"
    env = {"in_container": True, "has_sys_admin": False, "perf_event_open": 2}
    cat = classify_error(log, env)
    assert cat == "CONTAINER_MISSING_SYS_ADMIN"

def test_classifier_bare_metal_paranoid():
    log = "Error: perf_event_open failed with Permission denied"
    env = {"in_container": False, "has_sys_admin": True, "perf_event_open": 2}
    cat = classify_error(log, env)
    assert cat == "BARE_METAL_PERF_PARANOID"
