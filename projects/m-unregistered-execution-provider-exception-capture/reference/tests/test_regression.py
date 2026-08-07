import sys
sys.path.insert(0, ".")
from ortprovider.exceptions import safe_create_session, UnregisteredProviderError
from ortprovider.inference import compare_overhead

def test_unregistered_provider_raises_custom_exception():
    def bad_init():
        raise RuntimeError("Execution provider not available")
    try:
        safe_create_session(bad_init)
        assert False, "Expected UnregisteredProviderError"
    except UnregisteredProviderError:
        pass

def test_io_binding_overhead_reduction():
    class DummySession:
        pass
    metrics = compare_overhead(DummySession(), {})
    assert metrics.get("ratio", 1.0) < 1.0
