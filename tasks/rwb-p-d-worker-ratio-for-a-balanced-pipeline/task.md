## Context

A large language model serving system often separates work into two pipeline stages: prefill and decode.

Prefill workers process the input prompt tokens. If one prefill worker can process $r_p$ tokens per second and the mean input length is $L_p$ tokens, then $P$ prefill workers support an approximate request rate of

$$
\frac{P r_p}{L_p}.
$$

Decode workers generate output tokens. If one decode worker can process $r_d$ tokens per second and the mean output length is $L_d$ tokens, then $D$ decode workers support an approximate request rate of

$$
\frac{D r_d}{L_d}.
$$

A balanced pipeline chooses the smallest integer worker ratio $P:D$ such that both stages support the same request rate:

$$
\frac{P r_p}{L_p} = \frac{D r_d}{L_d}.
$$

Rearranging gives

$$
\frac{P}{D} = \frac{r_d L_p}{r_p L_d}.
$$

The ratio should be reduced to the smallest positive integer pair.

## Task

Implement `balanced_pd_ratio(prefill_tps_per_worker, decode_tps_per_worker, mean_input_len, mean_output_len)`.

The function receives four positive numbers:

- `prefill_tps_per_worker`: prefill token throughput of one worker.
- `decode_tps_per_worker`: decode token throughput of one worker.
- `mean_input_len`: average number of input tokens per request.
- `mean_output_len`: average number of generated output tokens per request.

Return a tuple `(P, D)` containing the smallest positive integers representing the balanced prefill-to-decode worker ratio.

## Example

```python
ratio = balanced_pd_ratio(
    prefill_tps_per_worker=4000,
    decode_tps_per_worker=200,
    mean_input_len=1000,
    mean_output_len=500,
)
# (1, 4)
```

The calculation is based on

$$
\frac{P}{D} = \frac{200 \cdot 1000}{4000 \cdot 500} = \frac{1}{10},
$$

so the example above would actually reduce to `(1, 10)`. The shown call demonstrates the inputs; the returned value must always be the reduced ratio from the formula.

## What the gate checks

The gate computes the reduced integer ratio using exact rational arithmetic from the balancing equation and compares the learner output against the oracle.

The `exact_match` score must equal $1.0$. Floating point approximations that produce a non-minimal ratio or ignore the input/output length mix will fail.
