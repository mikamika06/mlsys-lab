import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from gptqmem.model import simulate_timeline
    except ImportError:
        return {"timeline_match": 0.0}
    finally:
        sys.path.pop(0)

    ok = 0
    for cfg in ref.CONFIGS:
        want = ref.simulate_timeline(**cfg)
        try:
            got = simulate_timeline(**cfg)
            if got == want:
                ok += 1
        except Exception:
            pass

    return {"timeline_match": 1.0 if ok == len(ref.CONFIGS) else 0.0}
