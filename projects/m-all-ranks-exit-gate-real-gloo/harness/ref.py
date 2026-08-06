import time

def run_sync(workdir, rank, world_size, store_path, timeout_sec, queue):
    import sys
    sys.path.insert(0, workdir)
    try:
        from ft.gate import sync_or_exit
        start = time.time()
        res = sync_or_exit(rank, world_size, store_path, timeout_sec)
        duration = time.time() - start
        queue.put((rank, res, duration))
    except Exception as e:
        queue.put((rank, False, 0.0))
