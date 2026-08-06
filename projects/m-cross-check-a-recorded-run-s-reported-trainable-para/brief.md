# Ticket: Training run parameter audit discrepancies across fine-tuning clusters

**Reported By:** MLOps Efficiency Team
**Severity:** High (Auditing & Resource Allocation)

During recent fine-tuning experiments across various model architectures (Llama-3, Mistral, and custom internal transformer backbones), our experiment tracking database recorded inconsistent `trainable_parameters` numbers. Several runs using identical `LoraConfig` specifications reported vastly different trainable parameter counts when submitted across different worker nodes or framework wrappers.

For example, Run `run-7021` reported 13,631,488 trainable parameters, whereas Run `run-7022` with what should have been identical LoRA target modules and rank parameters reported 26,214,400 parameters. We suspect these discrepancies stem from inconsistencies in how adapter parameters, trainable bias terms, and extra saved modules (`modules_to_save`) are reported during run initialization.

Because billing and memory budget estimates rely on these reported numbers, we cannot trust historical run metadata until we have an automated auditing pipeline. We need a deterministic module that computes expected trainable parameters directly from `LoraConfig` and base model configurations, audits historical run records against these theoretical totals, and flags any runs with parameter count mismatches.
