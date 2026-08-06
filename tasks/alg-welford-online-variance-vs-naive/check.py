import numpy as np
import copy

def grade(sol, fx) -> dict:
    def reference(data):
        return float(np.var(np.array(data, dtype=np.float64)))

    my_fx = [
        {"data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        {"data": [1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0]},
        {"data": [1e12 + x for x in range(100)]},
        {"data": [1e10 + (x % 5) for x in range(1000)]}
    ]

    max_rel_err = 0.0
    count = 0

    for fixture in my_fx:
        data = fixture["data"]

        try:
            student_out = sol.welford_variance(copy.deepcopy(data))
            ref_out = reference(data)

            if ref_out == 0.0:
                err = abs(student_out - ref_out)
            else:
                err = abs(student_out - ref_out) / abs(ref_out)

            max_rel_err = max(max_rel_err, err)
            count += 1
        except Exception as e:
            return {"rel_err": float('inf')}

    if count == 0:
        return {"rel_err": float('inf')}

    return {"rel_err": float(max_rel_err)}
