import ref
import numpy as np


def check(workdir):
    from peft_mechanics.router import AdapterRouter

    out = {"outputs_matched": 0.0}

    base = {"q": np.array([[1.0, 2.0], [3.0, 4.0]])}
    cfg1 = {"peft_type": "LORA", "r": 1, "lora_alpha": 2, "target_modules": ["q"]}
    a1 = {"q": np.array([[1.0, 1.0]])}
    b1 = {"q": np.array([[1.0], [1.0]])}

    cfg2 = {"peft_type": "LORA", "r": 1, "lora_alpha": 1, "target_modules": ["q"]}
    a2 = {"q": np.array([[2.0, 2.0]])}
    b2 = {"q": np.array([[-1.0], [-1.0]])}

    x = np.array([1.0, 1.0])

    try:
        router = AdapterRouter(base)
        router.load_adapter("ad1", cfg1, a1, b1)
        router.load_adapter("ad2", cfg2, a2, b2)

        y1 = router.forward("q", x)

        router.set_adapter("ad2")
        y2 = router.forward("q", x)

        if np.allclose(y1, np.array([7.0, 11.0])) and np.allclose(y2, np.array([-1.0, 3.0])):
            out["outputs_matched"] = 1.0
        else:
            out["_note"] = f"outputs wrong: y1={y1}, y2={y2}"
    except Exception as e:
        out["_note"] = f"crashed: {e}"

    return out
