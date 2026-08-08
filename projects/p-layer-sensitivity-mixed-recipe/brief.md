We are preparing to deploy a latency-sensitive, memory-constrained model to an edge device. To fit within the device's strict RAM limits, we applied a uniform 4-bit quantization across all layers. Unfortunately, this naive approach resulted in a catastrophic 4% degradation in accuracy.

Upon closer inspection, uniform quantization harms the model because not all layers are equally sensitive. Large, over-parameterized layers can easily withstand 4-bit or even 2-bit compression, while small, critical bottleneck layers suffer massive precision loss that propagates through the network. We need a mixed-precision recipe.

Your task is to build an automated quantization allocator:
1. Implement a method to measure the isolated Mean Squared Error (MSE) degradation of quantizing each layer independently.
2. Build a greedy allocator that starts with maximum precision and iteratively downgrades the layer that offers the best trade-off (smallest MSE increase per byte saved) until the model fits the budget.
3. Compare the end-to-end MSE of your mixed-precision recipe against the best possible uniform precision recipe that fits the exact same memory budget.
4. Add a regression test to ensure the budget constraint is strictly honored.
