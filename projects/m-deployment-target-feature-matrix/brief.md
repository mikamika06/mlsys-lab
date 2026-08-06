# Ticket: Edge Export Failures on Target Deployment and Image Inputs

We are observing several intermittent issues during the export and conversion pipeline for CoreML deployment targets in our edge runner service.

First, when configuring models for older deployment targets (such as iOS 15 or iOS 16), the compilation phase occasionally fails because unsupported operations or features are inadvertently included in the exported graph without throwing a clear upstream warning. We need a robust mechanism to evaluate a deployment-target feature matrix and validate operator compatibility before final conversion.

Second, image-input conversion specifications are frequently breaking downstream because raw tensor inputs are passed where normalized image inputs with explicit scale, bias, and color space parameters are expected. When preprocessing metadata is misconfigured, models receive unscaled pixel tensors, leading to silent numerical degradation or runtime shape mismatch errors during inference on device.

Third, post-conversion inspection of the generated mlprogram packages lacks a clear inventory of the exact opset versions and operator frequencies emitted byெற்ற the compiler. Engineers cannot easily verify whether a model relies on deprecated functional blocks or new mlprogram constructs.

Please implement the required modules in exporttools/ to handle target feature validation, repair broken image input configurations, and enumerate mlprogram opsets accurately, accompanied by a comprehensive regression test suite in tests/test_regression.py.
