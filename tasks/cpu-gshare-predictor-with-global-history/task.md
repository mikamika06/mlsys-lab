## Context

A **gshare** branch predictor combines the global history register (GHR) with the program counter (PC) to index a pattern history table (PHT) of 2-bit saturating counters. The prediction is taken if the counter value $\ge 2$. After the branch resolves, the counter is updated: increment (sat at 3) if taken, decrement (sat at 0) if not taken, and the GHR is left-shifted by 1 with the outcome inserted.

Given a PC and the actual outcome (taken=1, not taken=0), the predictor's state evolves as:

$$\text{index} = (\text{PC} \oplus \text{GHR}) \bmod 2^k$$
$$\text{prediction} = \begin{cases} \text{taken} & \text{if PHT}[\text{index}] \ge 2 \\ \text{not taken} & \text{otherwise} \end{cases}$$
$$\text{PHT}[\text{index}] = \text{clamp}(\text{PHT}[\text{index}] + 2\cdot\text{outcome} - 1, 0, 3)$$
$$\text{GHR} = ((\text{GHR} \ll 1) \mathrel{|} \text{outcome}) \bmod 2^k$$

where $k$ is the history length / PHT size in bits (both equal).

## Task

Implement `simulate_branch(pc_list, outcome_list, k)`, which simulates a gshare predictor with $k$-bit GHR and PHT of $2^k$ entries (each a 2-bit saturating counter), initialised to 2 (weakly taken). The function returns the number of **mispredictions** (prediction $\neq$ actual outcome).

## Example

```python
>>> simulate_branch([0x100, 0x104], [1, 0], k=2)
1
```

Explanation: PC 0x100, GHR=0, index = 0x100 xor 0 mod 4 = 0, PHT[0]=2 → predict taken. Actual taken, correct. GHR becomes 1. Next PC 0x104, GHR=1, index = 0x104 xor 1 mod 4 = 0x101 mod 4 = 1, PHT[1]=2 → predict taken. Actual not taken → mispredict (+1). Final answer = 1.

## What the gate checks

`exact_match` = 1.0 if the returned mispredict count matches the reference implementation exactly, else 0.0.
