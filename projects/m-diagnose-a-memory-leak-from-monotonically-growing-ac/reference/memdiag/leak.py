def diagnose_leak(snapshots):
    if not snapshots or len(snapshots) < 2:
        return False
    bytes_list = [s["active_bytes"] for s in snapshots]
    return all(bytes_list[i] <= bytes_list[i+1] for i in range(len(bytes_list)-1)) and bytes_list[-1] > bytes_list[0]
