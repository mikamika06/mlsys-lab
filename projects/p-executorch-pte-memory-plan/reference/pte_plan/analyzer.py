def analyze_peak(pte_structure):
    tensors = pte_structure.get("tensors", [])
    max_peak = 0
    source_tensor = None
    for t in tensors:
        size = t.get("size", 0)
        if size > max_peak:
            max_peak = size
            source_tensor = t.get("name")
    return {"peak": max_peak, "source": source_tensor}

def split_program_data(pte_structure):
    tensors = pte_structure.get("tensors", [])
    program = []
    data = []
    for t in tensors:
        if t.get("type") == "weight":
            data.append(t)
        else:
            program.append(t)
    return {"program_tensors": program, "data_tensors": data}
