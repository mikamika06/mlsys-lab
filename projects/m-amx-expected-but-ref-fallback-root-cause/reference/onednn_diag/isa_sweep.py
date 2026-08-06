def analyze_k_sweep(sweep_records):
    sorted_records = sorted(sweep_records, key=lambda x: x['k'])
    transitions = []
    
    prev_isa = None
    for r in sorted_records:
        curr_isa = r['isa']
        if prev_isa is not None and curr_isa != prev_isa:
            transitions.append({
                'from_isa': prev_isa,
                'to_isa': curr_isa,
                'at_k': r['k']
            })
        prev_isa = curr_isa

    isa_counts = {}
    amx_times = []
    avx_times = []
    for r in sorted_records:
        isa = r['isa']
        isa_counts[isa] = isa_counts.get(isa, 0) + 1
        if 'amx' in isa.lower():
            amx_times.append(r['latency_ms'])
        elif 'avx' in isa.lower():
            avx_times.append(r['latency_ms'])

    dominant_isa = max(isa_counts.items(), key=lambda x: x[1])[0] if isa_counts else ""
    
    avg_amx = sum(amx_times) / len(amx_times) if amx_times else 0.0
    avg_avx = sum(avx_times) / len(avx_times) if avx_times else 0.0

    if avg_amx > 0 and avg_avx > 0:
        gain = (avg_avx - avg_amx) / avg_avx
    else:
        gain = 0.0

    return {
        'transitions': transitions,
        'dominant_isa': dominant_isa,
        'amx_efficiency_gain': round(gain, 4)
    }
