import sys
import numpy as np

def grade(sol, fx=None):
    # reference implementation of RoPE used as oracle
    def rope_ref(x, pos):
        B, S, H, D = x.shape
        freqs = 1.0 / (10000.0 ** (np.arange(0, D, 2, dtype=np.float64) / D))
        angles = pos[:, None].astype(np.float64) * freqs[None, :]
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)
        x_pairs = x.reshape(B, S, H, D // 2, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]
        cos_b = cos_vals[None, :, None, :]
        sin_b = sin_vals[None, :, None, :]
        out_even = x_even * cos_b - x_odd * sin_b
        out_odd = x_odd * cos_b + x_even * sin_b
        out_pairs = np.stack([out_even, out_odd], axis=-1)
        return out_pairs.reshape(B, S, H, D)

    # --- accuracy test ---
    np.random.seed(1234)
    shapes = [
        (2, 4, 2, 4),
        (3, 5, 6, 6),
        (1, 8, 1, 8),
        (4, 2, 3, 8),
    ]
    max_err = 0.0
    for B, S, H, D in shapes:
        x = np.random.randn(B, S, H, D).astype(np.float64)
        pos = np.arange(S, dtype=np.int64)
        try:
            out_student = sol.apply_rope(x.copy(), pos.copy())
        except Exception:
            max_err = 1.0
            break
        out_ref = rope_ref(x, pos)
        err = np.max(np.abs(out_student - out_ref))
        if err > max_err:
            max_err = err

    # --- line count test ---
    student_module_file = getattr(sol, '__file__', None)

    class LineCounter:
        def __enter__(self):
            self.count = 0
            self.old_trace = sys.gettrace()
            sys.settrace(self._trace)
            return self
        def __exit__(self, *args):
            sys.settrace(self.old_trace)
        def _trace(self, frame, event, arg):
            if event == 'line':
                if student_module_file and frame.f_code.co_filename == student_module_file:
                    self.count += 1
            return self._trace

    np.random.seed(42)
    B, S, H, D = 1, 8, 2, 6
    x_lc = np.random.randn(B, S, H, D).astype(np.float64)
    pos_lc = np.arange(S, dtype=np.int64)
    lc = LineCounter()
    try:
        with lc:
            out_lc = sol.apply_rope(x_lc, pos_lc)
        line_cnt = lc.count
    except Exception:
        line_cnt = 999999

    return {
        "max_abs_err": float(max_err),
        "line_count": line_cnt
    }
