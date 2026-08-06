import re

def parse_kernels(code: str) -> list[dict]:
    kernels = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("def triton_") or line.startswith("def cpp_"):
            m = re.match(r"def\s+([a-zA-Z0-9_]+)\s*\((.*?)\):", line)
            if m:
                name = m.group(1)
                args_raw = m.group(2).split(",")
                params = []
                for a in args_raw:
                    clean = a.split(":")[0].strip()
                    if clean and not clean.isupper() and "numel" not in clean:
                        params.append(clean)
                kernels.append({"name": name, "params": params})
    return kernels

def count_kernels(code: str) -> int:
    return len(parse_kernels(code))
