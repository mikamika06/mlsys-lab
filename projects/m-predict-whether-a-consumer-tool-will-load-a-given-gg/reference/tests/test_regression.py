import sys
sys.path.insert(0, ".")
from gguf_interop.compat import check_tool_compatibility

def test_compat_checks_required_keys():
    metadata = {
        "general.architecture": "llama",
        "llama.context_length": 4096
    }
    tool_profile = {
        "supported_architectures": ["llama"],
        "required_keys": ["{arch}.context_length", "{arch}.block_count"],
        "max_context_length": 8192
    }
    res = check_tool_compatibility(metadata, tool_profile)
    assert not res["compatible"]
    assert any("Missing required key" in r for r in res["reasons"])
