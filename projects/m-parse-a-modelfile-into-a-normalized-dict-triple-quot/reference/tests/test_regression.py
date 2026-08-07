import sys

sys.path.insert(0, ".")
from modelfile.parser import parse

def test_parser_keeps_all_stops():
    text = "FROM model\nPARAMETER stop \"<|im_end|>\"\nPARAMETER stop \"<|im_start|>\"\n"
    ast = parse(text)
    stops = ast.get("PARAMETER", {}).get("stop", [])
    assert len(stops) == 2, f"Expected 2 stops, got {len(stops)}"
    assert "<|im_end|>" in stops
    assert "<|im_start|>" in stops
