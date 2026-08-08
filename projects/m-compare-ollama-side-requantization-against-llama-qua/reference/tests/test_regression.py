import sys

sys.path.insert(0, ".")
from ollamabridge.quant import compare_requantization
from ollamabridge.import_model import verify_safetensors_dir
from ollamabridge.api import upload_and_create_model


def test_quant_comparison():
    res = compare_requantization({}, "Q4_K_M")
    assert "ollama_error" in res
    assert "llama_error" in res


def test_arch_verification():
    assert verify_safetensors_dir({"architecture": "llama"}) is True
    assert verify_safetensors_dir({"architecture": "unknown"}) is False


def test_api_upload():
    res = upload_and_create_model(b"test data", "my-model")
    assert res["status"] == "success"
    assert res["model"] == "my-model"
    assert res["digest"].startswith("sha256:")
