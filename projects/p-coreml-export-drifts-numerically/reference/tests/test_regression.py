import sys
import numpy as np

sys.path.insert(0, ".")
import ref
from coreml_export.converter import ModelConverter


def test_converter_output_shape():
    inputs = ref.generate_test_inputs(count=10)
    conv = ModelConverter()
    out = conv.predict(inputs)
    assert out.shape == (10, 3, 224, 224)


def test_converter_parity():
    inputs = ref.generate_test_inputs(count=50)
    conv = ModelConverter()
    out = conv.predict(inputs)
    ref_out = ref.run_reference_model(inputs)
    np.testing.assert_allclose(out, ref_out, rtol=1e-5, atol=1e-5)
