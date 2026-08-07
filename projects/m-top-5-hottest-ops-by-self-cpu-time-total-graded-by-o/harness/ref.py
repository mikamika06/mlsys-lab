PROFILES = [
    [
        {"Name": "aten::matmul", "Self CPU total": "10.5ms", "CPU total": "15.2ms", "Calls": 100},
        {"Name": "aten::add", "Self CPU total": "2.1ms", "CPU total": "2.1ms", "Calls": 500},
        {"Name": "aten::copy_", "Self CPU total": "45.0ms", "CPU total": "45.0ms", "Calls": 1000},
        {"Name": "aten::relu", "Self CPU total": "1.0ms", "CPU total": "1.5ms", "Calls": 200},
        {"Name": "aten::gelu", "Self CPU total": "3.5ms", "CPU total": "4.0ms", "Calls": 150},
    ],
    [
        {"Name": "aten::conv2d", "Self CPU total": "25.0ms", "CPU total": "30.0ms", "Calls": 50},
        {"Name": "aten::copy_", "Self CPU total": "60.0ms", "CPU total": "60.0ms", "Calls": 2000},
        {"Name": "aten::add", "Self CPU total": "5.0ms", "CPU total": "5.0ms", "Calls": 800},
        {"Name": "aten::mul", "Self CPU total": "4.0ms", "CPU total": "4.0ms", "Calls": 600},
        {"Name": "aten::silu", "Self CPU total": "8.0ms", "CPU total": "9.0ms", "Calls": 300},
    ]
]

def parse_table(rows):
    res = {}
    for r in rows:
        name = r["Name"]
        val_str = r["Self CPU total"]
        if val_str.endswith("ms"):
            val = float(val_str[:-2])
        elif val_str.endswith("us"):
            val = float(val_str[:-2]) / 1000.0
        else:
            val = float(val_str)
        res[name] = val
    return res

def top_k_ops(rows, k=5):
    parsed = parse_table(rows)
    sorted_ops = sorted(parsed.items(), key=lambda x: x[1], reverse=True)
    return [op for op, _ in sorted_ops[:k]]

def recall_at_k(ref_list, got_list, k=5):
    r_set = set(ref_list[:k])
    g_set = set(got_list[:k])
    if not r_set:
        return 1.0
    return len(r_set.intersection(g_set)) / len(r_set)
