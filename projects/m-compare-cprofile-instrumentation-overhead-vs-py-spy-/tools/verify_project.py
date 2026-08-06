#!/usr/bin/env python3
import os
import sys
import json

def check_milestone_1():
    target_file = "benchmark.py"
    if not os.path.exists(target_file):
        if os.path.exists("compare.py"):
            target_file = "compare.py"
        else:
            return False

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            content = f.read()
            return "cProfile" in content and ("time" in content or "profile" in content)
    except Exception:
        return False

def check_milestone_2():
    result_files = ["results.json", "benchmark_results.json", "data.json", "overhead.json"]
    for rf in result_files:
        if os.path.exists(rf):
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, (dict, list)) and len(data) > 0:
                        return True
            except Exception:
                if os.path.getsize(rf) > 10:
                    return True
    return False

def check_milestone_3():
    report_files = ["REPORT.md", "ANALYSIS.md", "COMPARISON.md"]
    for rf in report_files:
        if os.path.exists(rf):
            try:
                with open(rf, "r", encoding="utf-8") as f:
                    if len(f.read().strip()) > 50:
                        return True
            except Exception:
                pass
    return False

def verify_milestones(milestone_id):
    if milestone_id != "m-compare-cprofile-instrumentation-overhead-vs-py-spy-":
        print(f"Unknown milestone: {milestone_id}")
        sys.exit(1)

    m1 = check_milestone_1()
    m2 = check_milestone_2()
    m3 = check_milestone_3()

    milestones = [m1, m2, m3]
    cleared = sum(1 for m in milestones if m)
    total = len(milestones)

    if cleared == total:
        print(f"SUCCESS {milestone_id} reference clears {cleared}/{total}")
        sys.exit(0)
    else:
        print(f"FAIL {milestone_id} reference clears {cleared}/{total} (fails {total - cleared})")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tools/verify_project.py <milestone-id>")
        sys.exit(1)
    verify_milestones(sys.argv[1])
