import os
import tempfile
import multiprocessing as mp
import ref

def check(workdir):
    out = {"success_match": 0.0, "timeout_match": 0.0, "timing_measured": 0.0}
    
    with tempfile.TemporaryDirectory() as d:
        store_path = os.path.join(d, "store")
        q = mp.Queue()
        
        p0 = mp.Process(target=ref.run_sync, args=(workdir, 0, 2, store_path, 2.0, q))
        p1 = mp.Process(target=ref.run_sync, args=(workdir, 1, 2, store_path, 2.0, q))
        
        p0.start()
        p1.start()
        p0.join()
        p1.join()
        
        results = {}
        while not q.empty():
            rank, res, dur = q.get()
            results[rank] = (res, dur)
            
        if results.get(0, (None,))[0] is True and results.get(1, (None,))[0] is True:
            out["success_match"] = 1.0
            
    with tempfile.TemporaryDirectory() as d:
        store_path = os.path.join(d, "store2")
        q = mp.Queue()
        
        p0 = mp.Process(target=ref.run_sync, args=(workdir, 0, 2, store_path, 1.0, q))
        p0.start()
        p0.join(3.0)
        
        if p0.is_alive():
            p0.terminate()
            p0.join()
            out["_note"] = "Process hung instead of timing out"
        else:
            if not q.empty():
                rank, res, dur = q.get()
                if res is False:
                    out["timeout_match"] = 1.0
                if 0.5 <= dur <= 2.5:
                    out["timing_measured"] = 1.0
                if res is not False or not (0.5 <= dur <= 2.5):
                    out["_note"] = f"Result: {res}, Duration: {dur}"
            else:
                out["_note"] = "Queue empty, process crashed?"
    
    return out
