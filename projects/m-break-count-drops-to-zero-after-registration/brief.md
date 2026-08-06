# Custom Op TorchDynamo Integration and Graph Break Resolution

We recently noticed that our custom PyTorch tensor operator generates graph breaks during `torch.compile` passes, severely hampering kernel fusion and end-to-end execution performance across our training pipeline. The custom operation performs an elementwise transformation on incoming tensors, but PyTorch's dynamo compiler currently falls back to eager execution whenever it encounters the operator.

To fix this, we need to register the custom operator correctly using the modern `torch.library` API. However, registration alone might be insufficient or prone to silent failure if the operator schema or type signatures are invalid. We need to verify that the custom operation adheres strictly to PyTorch schema expectations via schema validation checks (`opcheck`). Once correctly registered and validated, graph breaks associated with this operator during `torch.compile` tracing should drop to zero.

Your goal is to inspect the existing unwrapped function, formalize its schema with `torch.library`, ensure it passes strict validation without schema violations, and confirm that `torch.compile` traces through it cleanly without graph breaks. Finally, write regression tests that verify both schema compliance and graph break prevention.
