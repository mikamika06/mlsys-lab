import numpy as np
from edgepipe.export import process_pipeline
from edgepipe.layout import diagnose_and_fix_layout
from edgepipe.quant import match_input_scale_zp


def test_layout_and_quantization_pipeline():
    raw_img = np.random.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    weights = np.random.randn(64, 3, 7, 7).astype(np.float32)
    bias = np.random.randn(64).astype(np.float32)

    fixed_img, (f_w, f_b) = process_pipeline(
        raw_img, "NHWC", "NCHW", "BGR", "RGB", mean, std, weights, bias
    )

    assert fixed_img.shape == (1, 3, 224, 224)
    scale, zp = match_input_scale_zp(mean, std)
    assert scale > 0.0
    assert 0 <= zp <= 255


def test_channel_reversal_integrity():
    raw_img = np.zeros((1, 10, 10, 3), dtype=np.uint8)
    raw_img[..., 0] = 255
    fixed = diagnose_and_fix_layout(raw_img, "NHWC", "NCHW", "BGR", "RGB")
    assert np.all(fixed[:, 2, :, :] == 255)
    assert np.all(fixed[:, 0, :, :] == 0)
