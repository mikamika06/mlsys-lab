# Ticket: KV Cache FP8 Calibration Failure with Llte-Compressor Scheme

We are observing severe perplexity regression and abnormal output discrepancies when deploying FP8 KV cache compression on our tiny evaluation models using the oneshot calibration flow.

## Symptoms Reported
- When running the standard oneshot KV cache recipe, the generated quantization scales for all layers evaluate uniformly to exactly `1.0`, disabling actual scaling and causing massive quantization distortion under low-bit FP8 representation.
- Downstream accuracy evaluation shows that uncalibrated or improperly scaled FP8 KV cache yields a much higher relative error (`rel_err`) compared to the full-precision baseline than expected, while proper calibration should tightly bound this error.
- Unit integration tests checking scale variation and accuracy delta currently pass on incorrect recipe outputs because the verification bounds are either missing or bypassed.

Please implement and verify the core modules under `compressor_kv/` to properly extract calibration statistics, repair constant-one scale recipes, and evaluate the calibrated vs uncalibrated FP8 KV cache relative error delta under rigorous test assertions.
