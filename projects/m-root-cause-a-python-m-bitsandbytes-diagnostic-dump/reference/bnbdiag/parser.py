import re

def parse_diagnostic(dump_text: str) -> dict:
    result = {
        "cuda_version": None,
        "platform": None,
        "lib_path": None,
        "error": None,
        "symbols": []
    }
    for line in dump_text.splitlines():
        line_s = line.strip()
        if line_s.startswith("CUDA version:"):
            result["cuda_version"] = line_s.split(":", 1)[1].strip()
        elif line_s.startswith("Platform:"):
            result["platform"] = line_s.split(":", 1)[1].strip()
        elif line_s.startswith("Library path:"):
            result["lib_path"] = line_s.split(":", 1)[1].strip()
        elif line_s.startswith("Error:"):
            result["error"] = line_s.split(":", 1)[1].strip()
        elif line_s.startswith("Symbol:"):
            result["symbols"].append(line_s.split(":", 1)[1].strip())
    return result
