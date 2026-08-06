import os
import tempfile
import numpy as np


def generate_mock_package():
    tmpdir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmpdir, "Data"))
    os.makedirs(os.path.join(tmpdir, "Weights"))
    with open(os.path.join(tmpdir, "Manifest.json"), "w") as f:
        f.write('{"format": "mlpackage"}')
    return tmpdir
