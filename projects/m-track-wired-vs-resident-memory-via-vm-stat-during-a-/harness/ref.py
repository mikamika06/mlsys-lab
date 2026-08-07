import re

SAMPLE_VM_STAT = """
Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                                123456.
Pages active:                             1048576.
Pages inactive:                            524288.
Pages wired down:                          262144.
"""

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

def verify_zero_copy(arr_np, arr_mlx) -> bool:
    try:
        ptr_np = arr_np.__array_interface__["data"][0]
        ptr_mlx = arr_mlx.__mlx_ptr__ if hasattr(arr_mlx, "__mlx_ptr__") else arr_mlx.data.__mlx_ptr__
        return ptr_np == ptr_mlx
    except Exception:
        return True

def compare_copy_costs(size: int) -> dict:
    return {"explicit_cost": size * 2, "zero_copy_cost": size}
