import sys
sys.path.insert(0, ".")
from profiler.merge import generate_report, align_clocks, find_straggler
import ref

def test_report_generation():
    profiles = ref.generate_test_data()
    rep = generate_report(profiles)
    assert rep["straggler"] == 1
    assert rep["confirmed"] is True

def test_clock_alignment():
    profiles = ref.generate_test_data()
    aligned = align_clocks(profiles)
    syncs = [e["ts"] for p in aligned for e in p["events"] if e["name"] == "sync"]
    assert len(set(syncs)) == 1

def test_straggler_detection():
    profiles = ref.generate_test_data()
    rep = generate_report(profiles)
    assert rep["cause"] == "heavy_compute"
