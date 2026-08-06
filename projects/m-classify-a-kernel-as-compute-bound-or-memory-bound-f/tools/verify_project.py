import sys
import importlib
import traceback

def verify_milestone(milestone_name):
    print(f"Verifying {milestone_name}...")
    try:
        if milestone_name == "m-classify-a-kernel-as-compute-bound-or-memory-bound-f":
            try:
                mod = importlib.import_module("kernel_classifier")
            except ImportError:
                try:
                    mod = importlib.import_module("classifier")
                except ImportError:
                    mod = None
            
            if mod and hasattr(mod, "classify_kernel"):
                # Test case 1: Memory bound (intensity < balance: 100/20 = 5 vs 1000/100 = 10)
                res1 = mod.classify_kernel({"flops": 100, "bytes": 20}, {"peak_flops": 1000, "peak_bandwidth": 100})
                if res1 is None:
                    raise AssertionError("classify_kernel returned None (unimplemented skeleton)")
                
                # Test case 2: Compute bound (intensity > balance: 200/10 = 20 vs 1000/100 = 10)
                res2 = mod.classify_kernel({"flops": 200, "bytes": 10}, {"peak_flops": 1000, "peak_bandwidth": 100})
                if res2 is None:
                    raise AssertionError("classify_kernel returned None (unimplemented skeleton)")
                
                def check_result(res, expected_type):
                    if isinstance(res, str):
                        if expected_type not in res.lower():
                            raise AssertionError(f"Expected classification to contain '{expected_type}', got '{res}'")
                    elif isinstance(res, dict):
                        found = any(isinstance(v, str) and expected_type in v.lower() for v in res.values())
                        if not found:
                            raise AssertionError(f"Expected dict to contain classification '{expected_type}', got {res}")
                    else:
                        raise AssertionError(f"Unexpected return type: {type(res)}")
                
                check_result(res1, "memory")
                check_result(res2, "compute")
            else:
                raise AssertionError("classify_kernel function not found")
        else:
            print(f"Unknown milestone: {milestone_name}")
            sys.exit(1)
            
        print(f"PASS {milestone_name}")
    except Exception as e:
        print(f"FAIL {milestone_name} {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_milestone(sys.argv[1])
    else:
        print("Usage: verify_project.py <milestone-name>")
        sys.exit(1)
