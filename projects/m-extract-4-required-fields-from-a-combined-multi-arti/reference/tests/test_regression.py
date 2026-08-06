import sys
import tempfile
import os
sys.path.insert(0, ".")
from extractor.parse import extract_required_fields
from extractor.decompile import verify_resume_signature
from extractor.debugdir import enumerate_debug_directory

def test_extraction_fields():
    log = "[TORCH_LOGS]: graph_id=1 node_count=5 op_name=aten::relu compile_status=SUCCESS"
    res = extract_required_fields(log)
    assert len(res) == 1
    assert res[0]["graph_id"] == "1"

def test_signature_verification():
    code = "def resume_func(x: int) -> int:\n    return x"
    assert verify_resume_signature(code, "(x: int) -> int") is True

def test_debug_directory_enumeration():
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "test.txt"), "w") as f:
            f.write("hello")
        files = enumerate_debug_directory(tmp)
        assert "test.txt" in files
