We are seeing sporadic NCCL timeouts during the All-to-All dispatch phase of our Mixture of Experts (MoE) training runs. The cluster operators reported that these deadlocks almost always happen right after the learning rate warmup phase finishes.

During this time, the router network suddenly becomes very confident, meaning the routing temperature drops effectively. This low temperature creates a massive load imbalance because tokens all clump to a few "popular" experts across the topology.

Because our All-to-All shape calculation isn't perfectly mapped to the actual dynamic routing assignments—especially when the capacity factor truncates the excess tokens—devices end up waiting for tokens that will never arrive, or trying to send tokens to an expert that has already closed its buffer, causing a complete cluster deadlock that requires a hard reset.

Your task is to fix this by accurately simulating the routing assignments under varying temperatures. First, implement temperature-scaled top-2 routing and calculate the load imbalance. Second, implement a search for the optimal capacity factor that bounds the dropped tokens while keeping the All-to-All shapes exact. Finally, output the exact send and receive token counts for each device partition.

To ensure this regression never creeps back in, add a strict safety test that verifies the `send_counts` matrix is the exact transpose of the `recv_counts` matrix.
