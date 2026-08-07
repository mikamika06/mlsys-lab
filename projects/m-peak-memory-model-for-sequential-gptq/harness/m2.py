import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from gptqmem.triage import triage_oom
    except ImportError:
        return {"triage_match": 0.0}
    finally:
        sys.path.pop(0)

    timelines = [ref.simulate_timeline(**cfg) for cfg in ref.CONFIGS]
    want = ref.triage_oom(timelines, ref.VRAM_LIMIT)
    try:
        got = triage_oom(timelines, ref.VRAM_LIMIT)
    except Exception:
        got = []

    return {"triage_match": 1.0 if got == want else 0.0}
