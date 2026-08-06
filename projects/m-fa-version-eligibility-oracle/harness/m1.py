import ref
from unittest.mock import patch

def check(workdir):
    from fa_oracle import stack
    out = {"stack_matched": 0.0, "configs": float(len(ref.ENVIRONMENTS))}
    ok = 0
    for i, env in enumerate(ref.ENVIRONMENTS):
        class MockTorch:
            __version__ = env["torch_version"]
            class version:
                cuda = env["cuda_version"]
            class cuda:
                @staticmethod
                def is_available():
                    return True
                @staticmethod
                def get_device_capability():
                    return env["compute_capability"]
                @staticmethod
                def get_device_name(i):
                    return env["device_name"]

        class MockFlashAttn:
            __version__ = env["installed_packages"].get("flash_attn")

        with patch("fa_oracle.stack.torch", MockTorch), \
             patch.dict("sys.modules", {"flash_attn": MockFlashAttn}):
            got = stack.detect_stack()
            want = ref.detect_stack_from_env(env)
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"env {i}: got {got}, want {want}"
    out["stack_matched"] = float(ok)
    return out
