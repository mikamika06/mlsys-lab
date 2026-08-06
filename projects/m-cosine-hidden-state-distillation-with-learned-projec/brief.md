# Feature Distillation Diagnostics: Hidden State Projection & Attention Map KL

Our internal transformer compression pipeline is encountering regression errors during student distillation from deep BERT-style teachers. Engineers report two main issues during intermediate-layer alignment:

1. Student hidden dimensions fail to align with teacher hidden dimensions due to direct MSE or improperly normalized cosine distance losses when applying linear projection layers.
2. Layer mapping and attention-map loss computation throw shape mismatches and non-finite KL divergence outputs across dynamic transformer depth configurations.

To resolve this, we need a standardized module for intermediate feature distillation that implements:
- Cosine distance loss for hidden states with a trainable linear projection layer for dimensional alignment.
- A deterministic layer-mapping generator following the TinyBERT strategy (mapping student layers to uniform intervals of teacher layers).
- A per-layer attention-map KL divergence metric function that computes distribution distances between teacher and student multi-head attention scores across layers safely and reproducibly.

Your task is to implement the core loss and mapping routines, verify their numerical accuracy, and write a test suite in `tests/test_regression.py` that catches common implementation mistakes, such as unnormalized cosine calculations or misaligned TinyBERT layer assignment tables.
