from collections import OrderedDict


def tier_lru(trace, sizes, savings, max_cpu_bytes):
    cpu = OrderedDict()
    used = 0

    for chunk in trace:
        if chunk in cpu:
            cpu.move_to_end(chunk)
        else:
            cpu[chunk] = None
            used += sizes[chunk]
            while used > max_cpu_bytes and cpu:
                old, _ = cpu.popitem(last=False)
                used -= sizes[old]

    resident = sorted(cpu.keys())
    total_savings = sum(savings[x] for x in resident)
    return resident, total_savings
