ASM_DICTS = [
    {"ptx": "hello", "cubin": b"world"},
    {"ttir": "tt.func", "llir": "define void"}
]

ASM_DICT_2 = {
    "ptx": ".version 7.5\n.target sm_80\n.entry kernel() {\n\tld.global.f32 %f1, [%rd1];\n\t// comment\n\tadd.f32 %f2, %f1, %f1;\n}\n",
    "cubin": b"x" * 1024
}

ASM_DICT_4 = {
    "ptx": ".version 7.5\n.target sm_80\n.entry kernel() {\n\tld.global.f32 %f1, [%rd1];\n\tld.global.f32 %f2, [%rd2];\n\tadd.f32 %f3, %f1, %f2;\n\t// comment\n\tadd.f32 %f4, %f3, %f1;\n}\n",
    "cubin": b"x" * 1536
}

SNIPPETS = [
    ".target sm_80\nld.shared.f32",
    "amdgcn-amd-amdhsa\ns_waitcnt vmcnt(0)",
    "gluon.backend.compile()",
    "generic snippet",
    "sm_90a",
    "PTX ISA"
]

def analyze_asm_dict(asm):
    return {k: len(v) for k, v in asm.items()}

def compare_num_stages(asm2, asm4):
    def count_inst(ptx):
        if not ptx: return 0
        cnt = 0
        for line in ptx.splitlines():
            line = line.strip()
            if not line or line.startswith('.') or line.startswith('//') or line.endswith(':'):
                continue
            cnt += 1
        return cnt

    p2 = asm2.get("ptx", "")
    p4 = asm4.get("ptx", "")
    return {
        "size_2": len(p2),
        "size_4": len(p4),
        "size_diff": len(p4) - len(p2),
        "inst_2": count_inst(p2),
        "inst_4": count_inst(p4),
        "inst_diff": count_inst(p4) - count_inst(p2)
    }

def classify_snippet(snippet):
    s = snippet.lower()
    if any(x in s for x in ("sm_", "ptx", ".target")):
        return "CUDA"
    if any(x in s for x in ("amdgcn", "s_waitcnt")):
        return "ROCm"
    if "gluon" in s:
        return "Gluon"
    return "Unknown"
