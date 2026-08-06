We are expanding our low-level fine-tuning analysis tools for Hugging Face linear layers and LoRA parameter scaling. Currently, our internal telemetry lacks precise verification of parameter counts across rank sweeps, dynamic target module expansion, and dropout stochasticity during adapter evaluation.

You need to build a modular LoRA parameter and behavior measuring toolkit that operates on model configs and tensor representations.

### What needs to be done

1. **Parameter Scaling & Rank Sweeps**: Implement rank sweep accounting for linear layers and target module selectors to measure real trainable parameter counts across different rank values $r$.
2. **Target Module Auto-Expansion**: Implement target layer auto-expansion matching Hugging Face PEFT's `target_modules='all-linear'` behavior, dynamically selecting all linear weight projections while skipping classifier/head modules.
3. **Adapter Dropout Stochasticity**: Measure output variance introduced by `lora_dropout` during training forward passes across evaluation iterations, verifying deterministic behavior when dropout is disabled vs stochastic variation when active.
4. **Safety Regression Suite**: Provide a regression test suite in `tests/test_regression.py` that verifies parameter accounting invariants and catches quiet fallbacks to all-module targeting or missing rank scaling.
