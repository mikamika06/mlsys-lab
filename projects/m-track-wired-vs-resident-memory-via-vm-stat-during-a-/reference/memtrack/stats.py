import re


def parse_vm_stat(output: str) -> dict:
    stats = {}
    for line in output.splitlines():
        if "Pages wired down" in line:
            m = re.search(r"(\d+)\.", line)
            if m:
                stats["wired"] = int(m.group(1)) * 4096
        elif "Pages active" in line:
            m = re.search(r"(\d+)\.", line)
            if m:
                stats["active"] = int(m.group(1)) * 4096
        elif "Pages inactive" in line:
            m = re.search(r"(\d+)\.", line)
            if m:
                stats["inactive"] = int(m.group(1)) * 4096
    return stats
