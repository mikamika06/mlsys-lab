from vllm_boot.server import boot_and_query_models


def test_served_model_name_returned_in_models_endpoint():
    args = [
        "--model",
        "facebook/opt-125m",
        "--served-model-name",
        "custom-opt-alias",
    ]
    res = boot_and_query_models(args)
    assert res["object"] == "list"
    assert len(res["data"]) == 1
    assert res["data"][0]["id"] == "custom-opt-alias"


def test_default_served_model_name():
    args = ["--model", "gpt2"]
    res = boot_and_query_models(args)
    assert res["data"][0]["id"] == "gpt2"
