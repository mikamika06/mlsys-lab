# Ticket: Diagnose Weight Distribution Where Q4_1 Beats Q4_0 and Fix Nibble Order Decoder

We are investigating a low-level quantization block issue in our llama.cpp legacy quant blocks integration. Specifically, we have a report regarding how the Q4_1 quantization format compares against Q4_0 under various weight distributions, and an intermittent data corruption bug suspected to stem from a wrong-nibble-order decoder implementation.

In llama.cpp, Q4_0 stores blocks of 32 weights with a 16-bit floating-point scale followed by 16 bytes of packed 4-bit integers (two 4-bit values per byte, with the lower or upper nibble packed in a specific endianness order). Q4_1 adds a 16-bit floating-point minimum/bias alongside the scale, changing the dynamic range and quantization error profile. For certain theoretical weight distributions (e.g., highly skewed distributions, asymmetric ranges, or specific kurtosis levels), Q4_1 surprisingly underperforms or outperforms Q4_0 depending on how the zero-point offset interacts with the variance.

Furthermore, our current low-level C/Python block decoder implementation is suffering from a subtle bug: when unpacking the 4-bit nibbles from the contiguous byte stream, the nibble order (high vs. low bits) is inverted on certain platforms or during block parsing, leading to severe decoding discrepancies and numerical degradation.

Your task is to implement the core analysis and decoding primitives:
1. Implement the mathematical evaluation to compute and find the specific weight distribution parameters where Q4_1 achieves a lower mean squared error (MSE) than Q4_0.
2. Implement the robust Q4 block encoder and decoder logic, ensuring that the nibble unpacking correctly handles the nibble order and resolves the decoder bug.
3. Write a comprehensive regression test suite in `tests/test_regression.py` that enforces correctness of the nibble decoder and guarantees that any inversion or tampering with the nibble extraction logic is successfully caught.
