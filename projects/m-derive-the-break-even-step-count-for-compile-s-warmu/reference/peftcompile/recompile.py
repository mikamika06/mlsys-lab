def count_adapter_recompilations(adapter_sequence):
    seen = []
    count = 0
    for adapter in adapter_sequence:
        if adapter not in seen:
            seen.append(adapter)
            count += 1
    return count
