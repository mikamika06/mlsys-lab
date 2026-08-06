import numpy as np

class QuadraticMemoryDetector:
    def analyze_allocations(self, execution_profile):
        flagged = []
        for op in execution_profile:
            seq_lens = np.array(op["seq_lengths"], dtype=np.float64)
            mem_bytes = np.array(op["peak_memory_bytes"], dtype=np.float64)
            if len(seq_lens) < 2:
                continue
            log_s = np.log(seq_lens)
            log_m = np.log(mem_bytes)
            poly = np.polyfit(log_s, log_m, 1)
            exponent = poly[0]
            is_quadratic = exponent >= 1.75
            flagged.append({
                "op_id": op["op_id"],
                "exponent": float(exponent),
                "is_quadratic": bool(is_quadratic)
            })
        return flagged
