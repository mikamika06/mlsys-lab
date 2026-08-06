import sys
sys.path.insert(0, ".")
from inspector.stages import compare_num_stages

def test_instruction_counting_ignores_directives_and_comments():
    asm2 = {"ptx": ".version 7.5\n// a comment\nlabel:\n\tadd.f32 a, b, c;\n"}
    asm4 = {"ptx": ".version 7.5\n// a comment\nlabel:\n\tadd.f32 a, b, c;\n\tadd.f32 c, d, e;\n"}
    res = compare_num_stages(asm2, asm4)
    assert res["inst_2"] == 1, f"Expected 1, got {res['inst_2']}"
    assert res["inst_4"] == 2, f"Expected 2, got {res['inst_4']}"
