def find_most_expensive_phase(annotated_data):
    totals = {}
    for item in annotated_data:
        name = item.get("name", "unknown")
        duration = item.get("end", 0) - item.get("start", 0)
        totals[name] = totals.get(name, 0) + duration
    if not totals:
        return None
    return max(totals, key=totals.get)

def analyze_gaps(trace_data):
    gaps = []
    sorted_items = sorted(trace_data, key=lambda x: x.get("start", 0))
    for i in range(len(sorted_items) - 1):
        end_current = sorted_items[i].get("end", 0)
        start_next = sorted_items[i+1].get("start", 0)
        if start_next > end_current:
            gaps.append({"start": end_current, "end": start_next, "duration": start_next - end_current})
    return gaps

def generate_phase_report(trace_data):
    report = {}
    for item in trace_data:
        name = item.get("name", "unknown")
        duration = item.get("end", 0) - item.get("start", 0)
        report[name] = report.get(name, 0) + duration
    return report

def verify_with_second_trace(trace1, trace2):
    r1 = generate_phase_report(trace1)
    r2 = generate_phase_report(trace2)
    p1 = max(r1, key=r1.get) if r1 else None
    p2 = max(r2, key=r2.get) if r2 else None
    return p1 == p2
