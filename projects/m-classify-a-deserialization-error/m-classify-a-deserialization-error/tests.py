import pytest
import json
import pickle
try:
    import yaml
except ImportError:
    yaml = None

from reference import classify_deserialization_error

def test_milestone_1_pickle_errors():
    """Milestone 1: Classify pickle and EOF/unpickling errors."""
    # Raise the exception directly instead of relying on pickle.loads
    # which can raise KeyError, ValueError, or TypeError depending on the Python version and payload.
    try:
        raise pickle.UnpicklingError("invalid load key")
    except Exception as e:
        res = classify_deserialization_error(e)
        assert res['format'] == 'pickle'
        assert res['category'] == 'unpickling_error'

    try:
        raise EOFError("Ran out of input")
    except Exception as e:
        res = classify_deserialization_error(e)
        assert res['format'] == 'pickle'
        assert res['category'] == 'eof'

def test_milestone_2_json_errors():
    """Milestone 2: Classify JSON decode errors with line/column details."""
    try:
        json.loads('{"invalid": json}')
    except Exception as e:
        res = classify_deserialization_error(e)
        assert res['format'] == 'json'
        assert res['category'] == 'json_decode_error'
        assert 'lineno' in res['details']

def test_milestone_3_yaml_or_unknown_errors():
    """Milestone 3: Classify YAML errors or fallback to unknown gracefully."""
    if yaml:
        try:
            yaml.safe_load('invalid: [unclosed')
        except Exception as e:
            res = classify_deserialization_error(e)
            assert res['format'] == 'yaml'
            assert res['category'] == 'yaml_error'

    # Unknown exception fallback
    try:
        raise ValueError("Some random error")
    except Exception as e:
        res = classify_deserialization_error(e)
        assert res['format'] == 'unknown'
        assert res['category'] == 'unknown'
