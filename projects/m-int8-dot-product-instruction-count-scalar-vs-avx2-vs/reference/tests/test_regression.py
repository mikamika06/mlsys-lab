import sys

sys.path.insert(0, ".")
from isa.parser import analyze_objdump


def test_strict_instruction_matching():
    # vpdpbusds is the saturating version, it shouldn't be counted as vpdpbusd
    lines = [
        "  40010: vpdpbusd %zmm0,%zmm1,%zmm2",
        "  40016: vpdpbusds %zmm0,%zmm1,%zmm2"
    ]
    res = analyze_objdump(lines)
    assert res["vpdpbusd"] == 1, f"Expected 1 vpdpbusd, got {res['vpdpbusd']}. Check for substring matching."
    assert res["vpaddd"] == 0
