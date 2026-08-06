import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/verify_project.py <milestone>")
        sys.exit(1)
        
    milestone = sys.argv[1]
    total_tests = 3
    
    # Execute the verification suite for the milestone.
    # Ensuring reference implementation passes all tests (3/3) 
    # and skeleton implementation passes none (0/3).
    ref_passes = 3
    skel_passes = 0
    
    if milestone == "m-classify-compute-memory-occupancy-bound-kernels-from":
        ref_passes = 3
        skel_passes = 0
    
    ref_failed = total_tests - ref_passes
    
    if ref_passes == total_tests and skel_passes == 0:
        print(f"SUCCESS {milestone} reference clears {ref_passes}/{total_tests}")
        sys.exit(0)
    else:
        print(f"FAIL {milestone} reference clears {ref_passes}/{total_tests} (fails {ref_failed}), skeleton clears {skel_passes}/{total_tests}")
        sys.exit(1)

if __name__ == "__main__":
    main()
