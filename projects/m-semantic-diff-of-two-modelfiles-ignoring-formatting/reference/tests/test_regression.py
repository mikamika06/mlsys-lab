import sys

sys.path.insert(0, ".")
from modelfile.diff import semantic_diff
from modelfile.validate import validate_modelfile
from modelfile.pin import verify_deterministic, build_deterministic_modelfile

def test_semantic_diff_ignores_formatting():
    mf1 = "FROM llama3\nPARAMETER temperature 0.7\n"
    mf2 = "  FROM   llama3  \n\n  PARAMETER   temperature   0.7  \n"
    diff = semantic_diff(mf1, mf2)
    assert not diff["added"] and not diff["removed"]

def test_validate_modelfile_catches_invalid_verb():
    mf = "FROM llama3\nBADINSTRUCTION foo\n"
    valid, line, err = validate_modelfile(mf)
    assert not valid
    assert line == 2

def test_verify_deterministic_pins():
    mf = build_deterministic_modelfile("llama3")
    assert verify_deterministic(mf) is True
