def find_missing_sync_points(trace_events):
    missing_indices = []
    stream_gpu_busy_until = {}

    for idx, event in enumerate(trace_events):
        event_type = event.get("type")
        stream_id = event.get("stream", 0)
        ts = event.get("ts", 0.0)
        dur = event.get("dur", 0.0)

        if event_type == "kernel":
            finish_time = ts + dur
            prev_finish = stream_gpu_busy_until.get(stream_id, 0.0)
            stream_gpu_busy_until[stream_id] = max(prev_finish, finish_time)
        elif event_type == "host_access":
            target_stream = event.get("target_stream", stream_id)
            busy_until = stream_gpu_busy_until.get(target_stream, 0.0)
            if busy_until > ts:
                missing_indices.append(idx)
        elif event_type == "synchronize":
            target_stream = event.get("target_stream", None)
            if target_stream is None:
                stream_gpu_busy_until.clear()
            else:
                stream_gpu_busy_until[target_stream] = 0.0

    return missing_indices
