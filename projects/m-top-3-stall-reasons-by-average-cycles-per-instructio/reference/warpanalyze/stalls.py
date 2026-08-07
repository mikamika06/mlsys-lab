def top_stall_reasons(warp_state_stats, k=3):
    ranked = []
    for entry in warp_state_stats:
        reason = entry["reason"]
        total_cycles = float(entry["total_stall_cycles"])
        total_insts = float(entry["total_executed_instructions"])
        avg_cpi = total_cycles / total_insts if total_insts > 0 else 0.0
        ranked.append((avg_cpi, total_cycles, reason))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [{"reason": r, "avg_cpi": cpi} for cpi, _, r in ranked[:k]]
