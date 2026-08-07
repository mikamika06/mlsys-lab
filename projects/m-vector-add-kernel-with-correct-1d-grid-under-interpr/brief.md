Subject: Non-power-of-two input tensors suffer from silent tail corruption in vector-add kernel

Our vector-add processing kernel behaves correctly when evaluation tensors have power-of-two lengths such as 1024 or 2048, matching reference PyTorch tensor addition element for element. However, during end-to-end integration tests on realistic request shapes (for example, tensor length `N=1000` with program `BLOCK_SIZE=128`), downstream assertion layers fail due to uninitialized NaN values persisting at the end of the output vector.

Debugging traces reveal two related issues. First, when grid launcher dimension calculations use simple integer division `N // BLOCK_SIZE`, the grid launches only 7 programs for 1000 elements, silently dropping the final 104 elements of the vector. Second, when attempting to launch 8 programs to cover the tail, unmasked memory writes extend past array boundaries or fail to properly handle boundary offsets under SPMD interpreter mode execution.

We need a clean 1D SPMD grid calculation utility that derives program count and quantifies launch waste, an accurate SPMD vector-add kernel with boundary masking, an under-launched grid simulation to reproduce and count silently dropped tail elements, and a regression test suite that catches grid dimensioning and boundary masking faults.
