# [BUG] Big-endian GGUF export corrupts metadata parsing and causes tensor misalignment on cross-platform runtimes

## Description
When attempting to export and reload quantized cross-platform GGUF model checkpoints for big-endian target systems, downstream inference runtimes encounter immediate memory corruption and header parse errors.

Specifically, metadata keys serialized into GGUF containers fail byte-order verification upon reloading. Additionally, when attempting zero-copy tensor mapping from memory-mapped GGUF files, tensor offsets are misaligned or shifted relative to the header padding boundary. This causes downstream matrix multiplications to read corrupt byte streams or throw unaligned memory access exceptions.

## Steps to Reproduce
1. Export a model configuration containing string, integer, array, and float KV metadata entries using big-endian serialization.
2. Read the resulting GGUF container using the container parser and verify metadata key-value restoration.
3. Attempt zero-copy tensor extraction using computed base offsets against memory-mapped array buffers.

## Expected Behavior
The container writer should construct a byte-exact big-endian GGUF binary container with proper 32-byte header padding. The container reader must correctly parse big-endian metadata and compute exact tensor base offsets, allowing zero-copy numpy array views without copying or byte swapping tensor payloads.
