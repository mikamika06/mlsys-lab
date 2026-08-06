We are observing numerical instability and incorrect output aggregation when scaling ring attention to long sequences across multiple simulated GPUs.

During context-parallel ring attention, partial attention outputs from different sequence chunks must be combined numerically using log-sum-exp (LSE) state tracking. In our current prototype, attention outputs across ring nodes show high relative error compared to exact single-device attention, and uneven casual masking creates significant load imbalances that degrade simulator throughput.

Your task is to implement the numerical LSE merging mechanism for partial attention blocks, build a single-process ring attention simulator, evaluate causal workload imbalances, and write a regression test suite to catch unsafe state tracking and improper LSE combination.

### Key Technical Focus
* Numerical stability in merging log-sum-exp ($m_i = \max(m_A, m_B)$) and updating weighted output vectors $O_{new} = \exp(m_A - m_{new}) \cdot O_A + \exp(m_B - m_{new}) \cdot O_B$.
* Single-process ring attention simulation maintaining state vectors across communication steps.
* Causal load imbalance metrics for ring attention iterations.
