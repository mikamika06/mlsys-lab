import sys
sys.path.insert(0, ".")
from runner.adapter import get_compatibility_matrix, fix_streaming_and_tokens

def test_compatibility_matrix_valid():
    mat = get_compatibility_matrix()
    assert isinstance(mat, dict)
    assert mat.get("chat_completions") is True

def test_streaming_fix():
    chunks = [{"delta": {"content": "hello world"}}]
    res, tokens = fix_streaming_and_tokens(chunks)
    assert len(res) == 1
    assert tokens > 0
