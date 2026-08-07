import sys
import ref

def check(workdir):
    out = {"benchmarks_complete": 0.0, "speedup_measured": 0.0}

    sys.path.insert(0, workdir)
    from gguf_pipeline.reporter import BenchmarkReporter

    reporter = BenchmarkReporter()
    reporter.add_entry("FP16", 100 * 1024 * 1024, 10.5, 0.00, 45.0)
    reporter.add_entry("Q8_0", 50 * 1024 * 1024, 10.6, 0.02, 75.0)
    reporter.add_entry("Q4_0", 25 * 1024 * 1024, 11.2, 0.15, 120.0)

    table = reporter.generate_table()
    if "Recipe" in table and "FP16" in table and "Q4_0" in table:
        out["benchmarks_complete"] = 1.0

    if len(reporter.entries) == 3 and reporter.entries[2]["tok_per_sec"] > reporter.entries[0]["tok_per_sec"]:
        out["speedup_measured"] = 1.0

    return out
