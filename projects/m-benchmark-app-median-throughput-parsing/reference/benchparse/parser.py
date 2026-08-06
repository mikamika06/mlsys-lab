import re

def parse_report(text):
    median_lat = None
    throughput = None
    hint = "latency"
    if "Throughput" in text or "throughput" in text.lower():
        hint = "throughput"
    
    m_lat = re.search(r"Median:\s+([0-9.]+)\s*ms", text)
    if m_lat:
        median_lat = float(m_lat.group(1))
    
    m_tp = re.search(r"Throughput:\s+([0-9.]+)\s*FPS", text)
    if m_tp:
        throughput = float(m_tp.group(1))
        
    return {"median_latency_ms": median_lat, "throughput_fps": throughput, "hint": hint}
