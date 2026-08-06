import sys
import numpy as np

sys.path.insert(0, ".")
from irhist.parser import parse_ir_xml
from irhist.histogram import build_histogram
from irhist.agreement import verify_numeric_agreement


SAMPLE_XML = """<net name="test_net" version="10">
    <layers>
        <layer id="0" name="data" type="Parameter"><output><port id="0" precision="FP32"><dim>1</dim><dim>3</dim></port></output></layer>
        <layer id="1" name="conv" type="Convolution"><input><port id="0"/></input><output><port id="1" precision="FP32"><dim>1</dim><dim>3</dim></port></output></layer>
        <layer id="2" name="relu" type="ReLU"><input><port id="0"/></input><output><port id="1" precision="FP32"><dim>1</dim><dim>3</dim></port></output></layer>
    </layers>
</net>"""


def test_parser_extracts_all_layers():
    ops = parse_ir_xml(SAMPLE_XML)
    assert len(ops) == 3
    assert ops[0]["type"] == "Parameter"
    assert ops[1]["type"] == "Convolution"
    assert ops[2]["type"] == "ReLU"


def test_histogram_counts_correctly():
    ops = parse_ir_xml(SAMPLE_XML)
    hist = build_histogram(ops)
    assert hist == {"Convolution": 1, "Parameter": 1, "ReLU": 1}


def test_numeric_agreement_passes_on_identical():
    a = [np.zeros((4, 4), dtype=np.float32)]
    b = [np.zeros((4, 4), dtype=np.float32)]
    assert verify_numeric_agreement(a, b, 1e-5, 1e-5) is True


def test_numeric_agreement_fails_on_mismatch():
    a = [np.ones((4, 4), dtype=np.float32)]
    b = [np.zeros((4, 4), dtype=np.float32)]
    assert verify_numeric_agreement(a, b, 1e-5, 1e-5) is False
