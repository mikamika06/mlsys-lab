# LoRA Hyperparameter and Parameter Scaling Issues in Model Fine-Tuning

Engineers on the fine-tuning team have reported several puzzling discrepancies when instrumenting LoRA adapter runs across transformer architectures.

First, when enabling adapter adapters for target modules using standard shorthand lists like `target_modules=['q_proj', 'v_proj']`, parameter tracking utilities are miscalculating total parameter counts compared to fully targeting all linear layers (`['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']`). A precise module resolution engine is needed to accurately resolve which actual nested PyTorch attribute paths correspond to specified module shorthands given a HuggingFace Transformer architecture structure.

Second, fine-tuning jobs targeting different rank values ($r$) produce wildly shifting gradient norms and learning dynamics when scaling $r$ without adapting alpha ($\alpha$). Naive $\alpha$-only scaling fails to preserve output variance across varying rank sizes.

Lastly, regression tests are needed to ensure adapter weight parameter counts and scaling behaviors hold invariant under architectural parameter shifts.

To fix these issues, you will build a LoRA inspection and parameter derivation library containing parameter counting tools, target module path resolvers, scaling law calculators, and safety regression tests.
