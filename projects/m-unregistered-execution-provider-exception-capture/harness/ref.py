import numpy as np
from ortprovider.exceptions import UnregisteredProviderError, safe_create_session
from ortprovider.inference import compare_overhead

def mock_create_fail():
    raise RuntimeError("Provider 'TensorrtExecutionProvider' is not available")

def mock_create_success():
    return "session_ok"
