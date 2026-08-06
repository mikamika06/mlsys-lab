def build_timeline(snapshot):
    traces = snapshot["device_traces"][0]
    timeline = []
    current_bytes = 0
    peak_bytes = 0
    
    for event in traces:
        if event["action"] == "alloc":
            current_bytes += event["size"]
        elif event["action"] == "free":
            current_bytes -= event["size"]
            
        if current_bytes > peak_bytes:
            peak_bytes = current_bytes
            
        timeline.append({"time": event["time"], "allocated_bytes": current_bytes})
        
    return timeline, peak_bytes
