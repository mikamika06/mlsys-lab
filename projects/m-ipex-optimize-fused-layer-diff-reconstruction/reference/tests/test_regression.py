import sys

sys.path.insert(0, ".")
from ipexdiff.diff import categorize_replacements, reconstruct_fused_diff


class MockSubModule:

    def __init__(self, name):
        self.__class__ = type(name, (object,), {})


class MockModel:

    def __init__(self, mod_dict):
        self._mods = mod_dict

    def named_modules(self):
        return self._mods.items()


def test_diff_reconstruction_detects_all_fused_modules():
    orig = MockModel(
        {
            "": MockSubModule("Module"),
            "layer1": MockSubModule("Linear"),
            "layer2": MockSubModule("Conv2d"),
            "layer3": MockSubModule("BatchNorm2d"),
        }
    )
    opt = MockModel(
        {
            "": MockSubModule("Module"),
            "layer1": MockSubModule("IpexLinear"),
            "layer2": MockSubModule("IpexConv2d"),
            "layer3": MockSubModule("BatchNorm2d"),
        }
    )

    diff = reconstruct_fused_diff(orig, opt)
    assert "layer1" in diff, "Failed to detect Linear -> IpexLinear replacement"
    assert "layer2" in diff, "Failed to detect Conv2d -> IpexConv2d replacement"
    assert "layer3" not in diff, "Incorrectly flagged unchanged BatchNorm2d"

    counts = categorize_replacements(diff)
    assert counts.get(("Linear", "IpexLinear")) == 1
    assert counts.get(("Conv2d", "IpexConv2d")) == 1
