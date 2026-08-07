# Edge Model Quantization Sizing and Operator Compatibility Audit

Our edge deployment pipeline for audio super-resolution models on embedded microcontrollers is failing on target devices. Production deployment scripts report that converted TFLite models are either exceeding the available flash budget or failing at runtime with missing tensor operator kernels.

The deployment team relied on simple weight-element count estimations to predict quantized flatbuffer sizes before running conversion. However, post-training quantization introduces per-tensor scale and zero-point overhead, custom metadata, alignment padding, and schema headers that cause actual `.tflite` sizes to deviate significantly from raw parameter estimates. Additionally, while dynamic-range quantization reduces parameter storage by quantizing weights to int8, it leaves activation tensors in float32, requiring full float32 runtime kernels. On the other hand, full-integer int8 and int16x8 post-training quantization modes convert activation pipelines to integer math, but impose strict op-level calibration constraints and distinct scale representation formats.

Your task is to analyze parameter size discrepancies, evaluate full-integer quantization variants across audio super-resolution layers, and provide regression tests to catch sizing and operational incompatibilities before deployment.
