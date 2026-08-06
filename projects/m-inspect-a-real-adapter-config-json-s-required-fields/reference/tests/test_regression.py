import sys
import numpy as np

sys.path.insert(0, ".")
from peft_mechanics.router import AdapterRouter


def test_switch_adapter_changes_output():
    base = {"layer1": np.eye(4)}
    router = AdapterRouter(base)
    config1 = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": ["layer1"]}
    config2 = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": ["layer1"]}

    a1 = {"layer1": np.ones((2, 4))}
    b1 = {"layer1": np.ones((4, 2))}

    a2 = {"layer1": np.full((2, 4), 2.0)}
    b2 = {"layer1": np.full((4, 2), 2.0)}

    router.load_adapter("ad1", config1, a1, b1)
    router.load_adapter("ad2", config2, a2, b2)

    x = np.ones(4)

    router.set_adapter("ad1")
    out1 = router.forward("layer1", x)

    router.set_adapter("ad2")
    out2 = router.forward("layer1", x)

    assert not np.allclose(out1, out2), "Output did not change after switching adapters"


def test_invalid_adapter_raises():
    base = {"layer1": np.eye(4)}
    router = AdapterRouter(base)
    try:
        router.set_adapter("missing")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
