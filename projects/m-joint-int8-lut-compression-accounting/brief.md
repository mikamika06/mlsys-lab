We're adding palettization and lookup-table compression (LUT) to our edge export tool. Initial reports claim 4-bit and 8-bit LUT compression per-channel saves massive memory, but our exported model files are sometimes LARGER than the float16 baseline!

After debugging, the issue is depthwise convolutions and other layers with very few input channels. The overhead of the per-channel LUT dominates. For example, a 4-bit per-channel palettization stores a 16-entry LUT for every single output channel. If those entries are `float16`, that's 32 bytes per channel just for the LUT. If a channel only has 9 weights, compressing it to 4-bit and adding a 32-byte LUT per channel is much worse than just storing 18 bytes for `float16`!

We need an exact accounting module `accounting/sizes.py` to compute memory usage in bytes for different methods:
- `"float16"`: 2 bytes per weight.
- `"int8_channel"`: 1 byte per weight, plus one `float16` scale (2 bytes) per output channel.
- `"lut4_channel_fp16"`: 4-bit indices packed per-channel (2 per byte, rounded up per channel), plus a 16-entry `float16` LUT (32 bytes) per output channel.
- `"lut4_joint_int8_channel"`: Joint int8+LUT. Same 4-bit indices, but the LUT itself is compressed to `int8`! So the LUT takes 16 bytes per channel, plus one `float16` scale (2 bytes) per channel (total 18 bytes/channel overhead).
- `"lut8_channel_fp16"`: 8-bit indices (1 byte each), plus a 256-entry `float16` LUT (512 bytes) per output channel.
- `"lut8_joint_int8_channel"`: 8-bit indices, but the LUT is `int8` (256 bytes) plus one `float16` scale (2 bytes) per channel (total 258 bytes/channel).

Implement `layer_bytes(shape, method)` and `optimize_model(shapes, allowed_methods)` to automatically pick the method that minimizes memory size.
