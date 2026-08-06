import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.ref as ref


def check(workdir):
    from serving.backend import configure_hopper_fa2, select_backend

    out = {"hopper_fa2_forced": 0.0, "non_hopper_rejected": 0.0}

    hopper_ok = True
    for hw in ref.get_hardware_profiles():
        if hw["is_hopper"]:
            env = {}
            try:
                res_env = configure_hopper_fa2(hw["name"], hw["cap"], env)
                backend = select_backend(hw["name"], hw["cap"], True, env)
                if res_env.get("VLLM_ATTENTION_BACKEND") != "FLASH_ATTN" or backend != "FLASH_ATTN":
                    hopper_ok = False
            except Exception as e:
                hopper_ok = False
                out["_note"] = f"Failed forcing FA2 on Hopper: {e}"
                break

    if hopper_ok:
        out["hopper_fa2_forced"] = 1.0

    rejected_ok = True
    for hw in ref.get_hardware_profiles():
        if not hw["is_hopper"]:
            try:
                configure_hopper_fa2(hw["name"], hw["cap"], {})
                rejected_ok = False
                out["_note"] = f"Non-Hopper hardware {hw['name']} was not rejected"
                break
            except ValueError:
                pass
            except Exception as e:
                rejected_ok = False
                out["_note"] = f"Unexpected exception on non-Hopper: {type(e).__name__}"
                break

    if rejected_ok:
        out["non_hopper_rejected"] = 1.0

    return out
