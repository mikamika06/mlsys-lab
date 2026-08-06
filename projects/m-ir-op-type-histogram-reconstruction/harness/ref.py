import numpy as np

NETS = [
    """<net name="net1" version="10"><layers>
        <layer id="0" name="in" type="Parameter"/>
        <layer id="1" name="relu1" type="ReLU"/>
    </layers></net>""",
    """<net name="net2" version="10"><layers>
        <layer id="0" name="in" type="Parameter"/>
        <layer id="1" name="conv" type="Convolution"/>
        <layer id="2" name="relu" type="ReLU"/>
    </layers></net>""",
    """<net name="net3" version="10"><layers>
        <layer id="0" name="in" type="Parameter"/>
        <layer id="1" name="matmul" type="MatMul"/>
        <layer id="2" name="add" type="Add"/>
        <layer id="3" name="relu" type="ReLU"/>
    </layers></net>"""
]


def generate_outputs(seed=42):
    rng = np.random.default_rng(seed)
    direct = [rng.normal(size=(8, 8)).astype(np.float32)]
    onnx = [direct[0] + rng.normal(scale=1e-7, size=(8, 8)).astype(np.float32)]
    return direct, onnx
