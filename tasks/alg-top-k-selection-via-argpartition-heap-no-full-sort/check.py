import sys
import numpy as np
import copy
import heapq

def grade(sol, fx) -> dict:
    def reference(arr, k):
        # Oracle: using numpy
        arr = np.array(arr)
        if k == 0: return []
        if k >= len(arr): return list(range(len(arr)))
        idx = np.argpartition(arr, k - 1)[:k]
        return set(idx.tolist())

    my_fx = [
        {"arr": [9.0, 1.0, 4.0, 7.0, 2.0, 5.0], "k": 3},
        {"arr": list(reversed(range(100))), "k": 10},
        {"arr": [1.0] * 50 + [0.0] * 5, "k": 5}
    ]

    correct = 1
    used_sort = 0
    
    for fixture in my_fx:
        arr = fixture["arr"]
        k = fixture["k"]
        
        # Trace execution to catch sort/sorted
        local_used_sort = False
        def trace_calls(frame, event, arg):
            nonlocal local_used_sort
            if event == 'c_call':
                func_name = getattr(arg, '__name__', '')
                if func_name in ('sort', 'sorted'):
                    local_used_sort = True
            return trace_calls
        
        try:
            sys.setprofile(trace_calls)
            student_out = sol.k_smallest_indices(copy.deepcopy(arr), k)
            sys.setprofile(None)
            
            ref_out = reference(arr, k)
            if set(student_out) != ref_out:
                correct = 0
                
            if local_used_sort:
                used_sort = 1
                
        except Exception as e:
            sys.setprofile(None)
            return {"correct": 0, "used_sort": 1}

    return {"correct": correct, "used_sort": used_sort}
