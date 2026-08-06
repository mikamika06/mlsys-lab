def find_retaining_frame(snapshot):
    traces = snapshot["device_traces"][0]
    live_allocs = {}
    
    for event in traces:
        if event["action"] == "alloc":
            live_allocs[event["addr"]] = (event["size"], event["frames"])
        elif event["action"] == "free":
            live_allocs.pop(event["addr"], None)
            
    frame_bytes = {}
    for size, frames in live_allocs.values():
        if frames:
            top_frame = f"{frames[-1]['filename']}:{frames[-1]['line']}:{frames[-1]['name']}"
            frame_bytes[top_frame] = frame_bytes.get(top_frame, 0) + size
            
    if not frame_bytes:
        return ("", 0)
        
    retaining_frame = max(frame_bytes.items(), key=lambda x: x[1])[0]
    total_retained = frame_bytes[retaining_frame]
    return retaining_frame, total_retained
