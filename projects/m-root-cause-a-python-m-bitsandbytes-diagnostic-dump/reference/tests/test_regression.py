import sys
sys.path.insert(0, ".")
from bnbdiag.resolver import resolve_shared_object
from bnbdiag.parser import parse_diagnostic
from bnbdiag.classifier import classify_traceback

def test_resolver_linux_x86_64():
    assert resolve_shared_object("12.1", "linux_x86_64") == "libbitsandbytes_cuda12.so"

def test_resolver_aarch64():
    assert resolve_shared_object("11.8", "linux_aarch64") == "libbitsandbytes_cuda11_sbs.so"

def test_parser_basic():
    dump = "CUDA version: 12.2\nPlatform: linux_x86_64\nError: libcudart missing\n"
    res = parse_diagnostic(dump)
    assert res["cuda_version"] == "12.2"
    assert res["platform"] == "linux_x86_64"

def test_classifier_basic():
    dump = {"error": "Failed to load cudart shared library"}
    assert classify_traceback(dump) == "MISSING_CUDART"
