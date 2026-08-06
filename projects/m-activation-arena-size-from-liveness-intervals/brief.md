# Ticket: Edge Runtime Memory Panic and Corrupted Constant Segment Offsets

## Problem Description

During edge deployment testing of exported vision and language models on microcontrollers and embedded NPUs running ExecuTorch runtime (.pte format), two critical runtime failures are observed.

First, the runtime fails during model loading with corrupted constant offset reads when parsing serialized `.pte` binaries. The loader fails to extract the constant segment offset and alignment properties, resulting in unaligned weights and misaligned DMA transfers during model setup.

Second, during inference execution, peak memory consumption in the tensor activation arena far exceeds the available static RAM budget, triggering runtime memory boundary panics. Analysis shows the current activation memory manager fails to reuse memory buffers across non-overlapping tensor liveness windows, leading to an over-allocated activation arena. Furthermore, without proper greedy-by-size allocation and byte-alignment packing, buffer fragmentation inflates the total arena size required.

We need a clean implementation of the `.pte` constant-segment header parser and a deterministic greedy-by-size memory planner driven by tensor liveness intervals to calculate the minimal activation arena size.
