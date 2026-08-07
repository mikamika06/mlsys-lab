import ref


def check(workdir):
    from profparse.parser import parse_events, map_tracks
    from profparse.timeline import compute_nesting_depth
    events, metadata, timestamps = ref.get_sample_data()
    parsed = parse_events(events)
    depths = compute_nesting_depth(parsed, timestamps)

    # expected depths for timestamps [50, 110, 150, 300] given forward[100, 250] and gemm[120, 180]
    # 50 -> 0
    # 110 -> 1 (forward active)
    # 150 -> 2 (forward and gemm active)
    # 300 -> 0
    want_depths = [0, 1, 2, 0]
    depth_matched = 1.0 if depths == want_depths else 0.0

    mapped = map_tracks(events, metadata)
    tracks_mapped = 1.0 if (len(mapped) > 0 and mapped[0].get("process_name") == "TrainerProcess" and mapped[0].get("thread_name") == "MainThread") else 0.0

    return {"depth_matched": depth_matched, "tracks_mapped": tracks_mapped}
