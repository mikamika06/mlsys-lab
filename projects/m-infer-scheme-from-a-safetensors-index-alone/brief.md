# Infer compressed-tensors scheme from safetensors index

Our checkpoint conversion pipeline received a batch of quantized model weights in compressed-tensors format. However, several downstream loader routines failed when attempting to restore these tensors. Upon investigation, we identified two critical metadata issues in the saved files:

First, several `safetensors` checkpoints only provide the tensor name mapping index (`model.safetensors.index.json`), but downstream allocation planning requires detecting the quantization scheme—specifically distinguishing between standard integer quantization (`int-quantized`) and packed integer quantization (`pack-quantized`)—from the index alone before reading large tensor files.

Second, multiple quantized tensor entries in the index are missing explicit `weight_shape` metadata attributes. Without `weight_shape`, memory planning and weight unpacking logic cannot correctly calculate the expected packed versus unpacked byte sizes or allocate destination buffers properly.

You need to fix our checkpoint inspection and metadata restoration utilities. Your tasks are:
1. Infer whether a checkpoint is using `int-quantized` or `pack-quantized` scheme based solely on the structural patterns in the `safetensors` index mapping.
2. Repair damaged checkpoint index records by inferring and inserting missing `weight_shape` metadata based on weight packing ratios, scale shapes, and packed byte dimensions.
3. Calculate precise byte footprint differences between `int-quantized` and `pack-quantized` tensor representations for arbitrary shapes and bit-widths.
4. Provide unit tests in `tests/test_regression.py` that validate repaired index metadata invariants and catch corrupted scale/shape assertions.
