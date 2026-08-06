# Ticket: Pre-Quantization Shape Validation for Importance Matrices

During the execution of imatrix-guided quantization pipelines for large language model weights in low-level serving frameworks, operators frequently experience silent quantization failures, unexpected memory access violations, or corrupted quantization weight scales when an importance matrix (`imatrix.dat`) generated from one architecture version or model scale is inadvertently applied to a target model with differing layer counts, hidden state dimensions, or attention head configurations.

Currently, the quantization toolchain attempts to map importance data entries directly by string matching tensor names while completely bypassing rigorous validation of underlying tensor dimensions and matrix shapes. This results in misaligned quantization scales and silent numerical degradation during subsequent inference tasks.

We require a robust pre-quantization validation utility module that inspects model tensor shape configurations against loaded imatrix metadata structures prior to initiating quantization routines. This module must accurately verify individual tensor dimension compatibility, generate detailed mismatch and omission reports across the entire model tensor catalog, and provide comprehensive regression test suites to guarantee that shape-mismatch faults are caught reliably before writing output files.
