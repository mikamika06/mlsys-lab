## Context

In distributed training with Fully Sharded Data Parallel (FSDP), parameters and activations may reside on the GPU or be offloaded to CPU depending on flags such as `cpu_offload`, `activation_checkpoint`, and `activation_offload`. The residency of each component during the forward and backward passes determines memory usage and performance.

## Task

Implement a function

```python
def residency(cpu_offload: bool,
              activation_checkpoint: bool,
              activation_offload: bool) -> Dict[str, List[str]]:
    ...
```

that returns a mapping from phase names (`"forward"` and `"backward"`) to lists of items that are guaranteed to be on the GPU during that phase. The possible items are:

- `shard` – the sharded portion of the parameters  
- `full_param` – the full parameter tensor (when not sharded)  
- `activations` – the activations produced by the forward pass  

The rules for residency are:

1. If `cpu_offload` is `True`, both `shard` and `full_param` are offloaded to CPU during *both* phases.  
2. If `activation_checkpoint` is `True`, activations are not stored after the forward pass; they are recomputed on GPU during backward, so they appear only in the `"backward"` list.  
3. Otherwise, if `activation_offload` is `True`, activations are offloaded to CPU for *both* phases and never appear in either list.  
4. If none of the above conditions apply, activations stay on GPU in both phases.

The function must return lists sorted alphabetically to ensure deterministic comparison.

## Example

```python
>>> residency(False, False, False)
{'forward': ['activations', 'full_param', 'shard'],
 'backward': ['activations', 'full_param', 'shard']}

>>> residency(True, True, False)
{'forward': [],
 'backward': ['activations']}
```

## What the gate checks

The grader computes an oracle mapping using the same rules and compares it to your output with exact match. Any deviation or exception causes failure.
