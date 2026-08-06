import re

def analyze_throughput(isa: str, vec_bits: int, ports: int) -> dict:
    if isa == "scalar":
        elements = 1
        insts = 2
    elif isa == "avx2":
        elements = vec_bits // 8
        insts = 3
    elif isa == "avx512_vnni":
        elements = vec_bits // 8
        insts = 1
    else:
        raise ValueError(f"Unknown isa: {isa}")

    return {
        "elements_per_vec": elements,
        "instructions_per_sequence": insts,
        "macs_per_cycle": round((elements / insts) * ports, 2)
    }


def analyze_objdump(lines: list[str]) -> dict:
    counts = {"vpmaddubsw": 0, "vpmaddwd": 0, "vpaddd": 0, "vpdpbusd": 0}
    for line in lines:
        words = re.findall(r'[a-zA-Z0-9_]+', line.lower())
        for w in words:
            if w in counts:
                counts[w] += 1
    return counts


def generate_objdumps():
    return {
        "avx2_loop": [
            "401000: c4 e2 7d 04 d0 vpmaddubsw %ymm0,%ymm1,%ymm2",
            "401005: c4 e2 6d 04 da vpmaddwd %ymm2,%ymm3,%ymm3",
            "40100a: c5 e5 fe c3 vpaddd %ymm3,%ymm4,%ymm0",
            "40100e: e2 f0 loop 401000"
        ],
        "vnni_loop": [
            "402000: 62 f2 75 48 50 d0 vpdpbusd %zmm0,%zmm1,%zmm2",
            "402006: e8 00 00 00 00 callq 402011 <vpdpbusd_handler>",
            "40200b: e2 f8 loop 402000"
        ],
        "tricky_saturating": [
            "403000: 62 f2 75 48 51 d0 vpdpbusds %zmm0,%zmm1,%zmm2",
            "403006: 62 f2 75 48 50 d0 vpdpbusd %zmm0,%zmm1,%zmm3",
            "vpdpbusdw is not a real instruction but let's test it"
        ]
    }
