Our Android edge deployment pipeline has run into a severe performance regression on several target devices when attempting to offload model execution to hardware accelerators using custom delegates. During integration testing of our latest vision model variant, we observed that while the delegation interface successfully registers without crashing, inference throughput completely collapses and execution falls back entirely to the CPU runtime, or results in a complete refusal to partition any operations.

Engineering has isolated the problem space down to three critical requirements that must be systematically addressed within our delegate integration package:

First, we need a robust delegate support classifier capable of inspecting a target model graph and accurately evaluating support status for exactly fifteen core operations across varying tensor dimensions and data types. Without precise capability filtering, the runtime attempts invalid offloads that trigger silent aborts deep inside the native driver layers.

Second, when delegation fails entirely, producing zero delegated nodes, our diagnostic tooling currently provides no actionable insight. We need a zero-node delegation postmortem generator that analyzes graph mismatch reasons, unsupported op signatures, and attribute constraints, producing a structured breakdown explaining precisely why fallback occurred.

Third, splitting a computation graph between the CPU and a hardware accelerator introduces substantial overhead at partition boundaries. We need a partition-boundary copy cost model that accurately computes tensor serialization, memory transfer, and layout conversion overheads to determine whether a proposed graph cut yields a net performance gain or merely incurs prohibitive inter-device transfer penalties.

Your task is to implement the delegate support classifier, the zero-node delegation postmortem analyzer, and the partition-boundary copy cost model, accompanied by a comprehensive test suite that catches regressions in boundary cost evaluations.
