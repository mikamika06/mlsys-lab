# Pick the calibration method for ORT quantization

A deployment pipeline for a transformer model converted to ONNX is experiencing severe accuracy degradation when switching from FP16 to INT8 static quantization using ONNX Runtime (ORT). The downstream team reports that setting fixed symmetric ranges causes output logits to drift, while naive min-max calibration produces severe scale collapse on attention layers with heavy outliers.

You are tasked with building a calibration diagnostic and quantization configuration selector (`calib/picker.py`, `calib/collapse.py`, and `calib/schema.py`). The pipeline needs to inspect activation histograms, detect tensor scale collapse across calibration methods, choose between S8S8 (signed 8-bit activation and weight) and U8S8 (unsigned 8-bit activation, signed 8-bit weight) schemes depending on activation symmetry, and recommend optimal calibration parameters (MinMax, Entropy, or Percentile).

Finally, you must write a regression test suite in `tests/test_regression.py` that verifies calibration selection invariants under distribution shifts and outlier spikes.
