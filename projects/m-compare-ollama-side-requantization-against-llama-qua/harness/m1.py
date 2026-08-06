import os
import tempfile
import ref


def check(workdir):
    out = {"imports_checked": 0.0}
    import sys
    sys.path.insert(0, workdir)
    try:
        from runner.import_tools import (
            convert_safetensors_to_gguf_manifest,
            inspect_and_verify_safetensors,
        )
    except Exception as e:
        out["_note"] = f"Failed to import runner.import_tools: {e}"
        return out

    with tempfile.TemporaryDirectory() as tmpdir:
        dirs = ref.setup_test_directories(tmpdir)

        res1 = inspect_and_verify_safetensors(dirs["valid"], ref.SUPPORTED_ARCHS)
        if not (res1.get("supported") is True and res1.get("architecture") == "LlamaForCausalLM"):
            out["_note"] = f"Valid directory failed check: {res1}"
            return out

        res2 = inspect_and_verify_safetensors(dirs["unsupported"], ref.SUPPORTED_ARCHS)
        if res2.get("supported") is not False or res2.get("reason") != "unsupported_architecture":
            out["_note"] = f"Unsupported directory check failed: {res2}"
            return out

        res3 = inspect_and_verify_safetensors(dirs["no_weights"], ref.SUPPORTED_ARCHS)
        if res3.get("supported") is not False or res3.get("reason") != "missing_safetensors_files":
            out["_note"] = f"Missing weights check failed: {res3}"
            return out

        res4 = inspect_and_verify_safetensors(dirs["missing_config"], ref.SUPPORTED_ARCHS)
        if res4.get("supported") is not False or res4.get("reason") != "missing_config":
            out["_note"] = f"Missing config check failed: {res4}"
            return out

        manifest = convert_safetensors_to_gguf_manifest(dirs["valid"])
        if manifest.get("general.architecture") != "llama" or manifest.get("embedding_length") != 2048:
            out["_note"] = f"Manifest conversion incorrect: {manifest}"
            return out

        out["imports_checked"] = 4.0

    return out
