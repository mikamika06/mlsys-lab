# Target Modules Semantics Verification

A fine-tuning pipeline for LoRA configuration recently failed silently in production when adapting a transformer backbone. The engineering team noticed that specifying `target_modules` as a list of leaf string suffixes vs. a single regular expression string appeared to target different sets of PyTorch linear modules, leading to mismatched adapter application and degraded convergence.

Your job is to inspect the model's named module hierarchy and implement a verification tool that determines whether a target module regex pattern and a target module suffix list wrap the exact same set of named submodules within a given PyTorch model structure.

You will implement the target module resolver, compare regex vs. suffix list target specifications against model module trees, evaluate parameter counts affected, and author a regression test suite to prevent divergent target module selection in future training runs.
