import sys
sys.path.insert(0, ".")
from jsonfsm.fsm import JSONFSM
from jsonfsm.mask import compute_mask, verify_equivalence
from jsonfsm.schema import detect_unsupported_keywords
import numpy as np

def test_fsm_basic():
    vocab = ["{", '"key"', ":", '"val"', "}", "foo"]
    fsm = JSONFSM({})
    assert fsm.step("{") == "OBJECT_KEY"
    allowed = fsm.allowed_tokens(vocab)
    assert 0 not in allowed

def test_mask_equivalence():
    logits = np.array([1.0, 5.0, 2.0, 3.0])
    allowed = [1, 3]
    masked = compute_mask(logits, allowed)
    assert verify_equivalence(logits, masked, allowed)

def test_unsupported_schema():
    schema = {"type": "object", "$ref": "#/definitions/Foo"}
    unsupported = detect_unsupported_keywords(schema)
    assert "$ref" in unsupported
