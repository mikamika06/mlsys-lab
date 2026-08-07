import ref

def check(workdir):
    m = {"peak_match": 0.0, "source_match": 0.0}
    try:
        from pte_plan.parser import parse_pte
        from pte_plan.analyzer import get_peak_memory
        data = ref.generate_pte_artifact()
        parsed = parse_pte(data)
        peak, source = get_peak_memory(parsed)
        ref_peak, ref_source = ref.expected_peak_and_source(parsed)
        if peak == ref_peak:
            m["peak_match"] = 1.0
        if source == ref_source:
            m["source_match"] = 1.0
    except Exception:
        pass
    return m
