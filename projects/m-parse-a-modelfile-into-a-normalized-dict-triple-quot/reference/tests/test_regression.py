import sys
sys.path.insert(0, ".")
from modelfile.parser import parse_modelfile
from modelfile.emitter import emit_modelfile

SAMPLE = (
    "FROM llama3\n"
    'PARAMETER stop "</s>"\n'
    'PARAMETER stop "<|eot_id|>"\n'
    'SYSTEM """\nHello\n"""\n'
)

def test_repeated_stop_preserved():
    parsed = parse_modelfile(SAMPLE)
    assert isinstance(parsed["parameters"].get("stop"), list)
    assert len(parsed["parameters"]["stop"]) == 2

def test_roundtrip_stability():
    parsed1 = parse_modelfile(SAMPLE)
    emitted = emit_modelfile(parsed1)
    parsed2 = parse_modelfile(emitted)
    assert parsed1 == parsed2
