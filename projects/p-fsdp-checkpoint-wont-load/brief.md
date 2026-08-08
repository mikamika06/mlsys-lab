We spent the entire last week training our new embedding model on a cluster with 8 GPUs using Fully Sharded Data Parallel. We saved the checkpoint successfully and shut down the instance over the weekend to save costs.

Today we tried to resume the training to push it for a few more epochs, but the orchestrator gave us an allocation of 12 GPUs instead of 8. When we try to load the checkpoint, the training script crashes immediately with a shape mismatch error. It seems the checkpoint format is hardcoded to the number of ranks it was saved with, because each file just contains a sliced array instead of the real parameters.

That checkpoint cost a week of compute. We absolutely cannot afford to start over from scratch.

We need a utility to read these sharded state directories, figure out the actual structure, and convert them into a consolidated, portable format so we can resume training on any number of GPUs. We also need to be completely sure that the math remains identical. We cannot have a bug in the reshaping logic that silently degrades the weights while training resumes. Build something that proves the evaluation loss on a dummy batch remains exactly the same before and after the checkpoint format transformation.
