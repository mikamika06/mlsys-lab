# Diagnostic & Performance Audit: EAGLE-3 + Tensor-Parallel Speculative Decoding

Production serving clusters running tensor-parallel (TP) inference with EAGLE-3 speculative decoding exhibit unpredictable aggregate goodput and silent startup failures during scaling. In several deployments across vLLM, SGLang, and TensorRT-LLM runtimes, systems either fail to start cleanly or produce lower aggregate goodput under speculative decoding than baseline non-speculative serving.

You are tasked with building an analytical log-replay auditor and diagnostic framework to resolve these issues across two primary pillars:

1. **Startup Diagnosis**: Analyze logged tensor-parallel worker handshake frames and EAGLE-3 initialization vectors to detect and categorize 6 distinct startup failures:
   - `TP_RANK_MISMATCH`: Rank ID set inconsistent across communication ring nodes.
   - `DRAFT_HEAD_SHAPE_MISMATCH`: Draft model hidden dimension does not match base model output projection.
   - `IPC_MEM_HANDLE_LEAK`: Unclosed IPC shared memory handles from aborted rank initializations.
   - `TREE_MASK_BUFFER_OVERFLOW`: EAGLE-3 tree mask buffer size smaller than draft tree width x max depth.
   - `VOCAB_SIZE_OUT_OF_BOUNDS`: Speculative proposal head vocabulary dimension mismatches target base LM vocabulary.
   - `NCCL_TIMEOUT_DEADLOCK`: Asynchronous collective barrier deadlock across TP sockets.

2. **Aggregate Goodput Evaluation**: Compute and compare system aggregate goodput from recorded scheduler trace logs under standard vs speculative decoding. Goodput must account for token acceptance rates, speculation draft lengths, latency target SLAs, and rejection overhead penalties.

3. **Regression Safeguard**: Implement test suites in `tests/test_regression.py` that verify goodput aggregation calculations and catch naive estimation bugs (such as ignoring speculative rejection penalty multipliers or latency SLA cutoffs).
