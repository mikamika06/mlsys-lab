# Symptom

Production deployments are encountering mysterious engine deserialization failures. We distribute TensorRT `.plan` files to various edge and datacenter nodes, but version mismatches, OS differences, and GPU architecture incompatibilities are causing crashes at load time.

Furthermore, we recently enabled "Hardware Compatibility" when building some of our engines. This feature allows an engine compiled for one architecture (like Ampere SM 80) to run on another (like Ada SM 89), which is great for portability. However, users are reporting two issues:
1. Sometimes these "compatible" engines still fail to deserialize on older cards.
2. Even when they succeed, they exhibit a silent performance penalty due to the runtime relying on generic cross-architecture kernels instead of highly tuned SM-specific code.

# Request

We need you to implement a robust deserialization diagnostic tool in pure Python. You will parse the 20-byte binary header of our `.plan` files to extract the magic string, TRT version, build SM architecture, hardware compatibility flag, and OS platform.

Then, write a diagnostic routine `diagnose_load` that checks this header against the target environment's specs. It must return a specific error classification if the load will fail, and calculate the expected performance penalty percentage (if any) when the engine successfully loads using fallback kernels.

Finally, write a regression test to ensure that the hardware compatibility mode strictly enforces its underlying requirement: it only supports Ampere (SM 80) and newer architectures.
