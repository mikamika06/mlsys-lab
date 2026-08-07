import os
import tempfile
import ref


def check(workdir):
    m = {"imatrix_generated": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        calib = os.path.join(tmp, "calib.txt")
        with open(calib, "w") as f:
            f.write("calibration text")
        out_mat = os.path.join(tmp, "imatrix.dat")

        import sys
        sys.path.insert(0, workdir)
        try:
            import gguf_pipe.quantize as quant
            quant.generate_imatrix("model.gguf", calib, out_mat)
            if os.path.exists(out_mat):
                m["imatrix_generated"] = 1.0
        except Exception:
            pass
    return m
