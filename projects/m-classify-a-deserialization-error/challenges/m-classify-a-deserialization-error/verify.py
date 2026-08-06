import json
import pickle
try:
    import yaml
except ImportError:
    yaml = None

from solution import classify_deserialization_error as ref_solve
from skeleton import classify_deserialization_error as skel_solve

def test_milestone_1():
    err = json.JSONDecodeError("Expecting value", "", 0)
    assert ref_solve(err) == "json_error"
    assert skel_solve(err) != "json_error"

def test_milestone_2():
    err = pickle.UnpicklingError("invalid load key")
    assert ref_solve(err) == "pickle_error"
    assert skel_solve(err) != "pickle_error"

def test_milestone_3():
    if yaml:
        err = yaml.YAMLError("parser error")
        assert ref_solve(err) == "yaml_error"
        assert skel_solve(err) != "yaml_error"
    else:
        assert True
