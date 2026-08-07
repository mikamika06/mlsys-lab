We need a robust suite of token filtering routines for our local inference runner. Currently, models have a tendency to babble or repeat themselves, and users are requesting finer control over the generation characteristics.

Specifically, we want you to implement the following sampling filters from scratch in NumPy:
1. `apply_repetition_penalty`: Penalizes logits of recently generated tokens over a rolling `repeat_last_n` window.
2. `apply_top_k`: Keeps only the top K tokens, masking out the rest to `-inf`.
3. `apply_top_p`: Keeps the minimal set of highest-probability tokens whose cumulative probability exceeds P.
4. `apply_min_p`: A newer method that removes tokens whose probability is less than `P` times the *maximum* probability token.
5. `apply_temperature`: Scales the logits.

Then, chain these all together in `full_chain`, applying them in this exact order: Repetition Penalty -> Top-K -> Top-P -> Min-P -> Temperature.

Finally, write a small function `compare_survival(logits, top_p, min_p)` that returns a tuple of sets: `(tokens_surviving_top_p, tokens_surviving_min_p)` to let us analyze which tokens each algorithm permits on a given probability distribution.

We are seeing bugs in alternative implementations of min-p where folks forget it is relative to the *maximum probability* and treat it like a flat absolute probability threshold. Please include a regression test in `tests/test_regression.py` ensuring `apply_min_p` correctly scales relative to the top probability.
