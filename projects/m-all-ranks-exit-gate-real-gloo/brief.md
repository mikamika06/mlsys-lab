Our distributed training jobs have been experiencing two major reliability issues related to fault tolerance that we need you to fix. 

First, when a node goes down during a run, the surviving nodes often hang indefinitely in a collective operation, wasting expensive GPU hours. We suspect the PyTorch timeout isn't catching properly, or we are not exiting ranks cleanly on failure. We need an "All-Ranks-Exit Gate"—a robust synchronization barrier backed by PyTorch's Gloo backend using a FileStore. It must attempt to synchronize all ranks, enforcing a strict timeout. If a timeout (or any exception) occurs, it should safely clean up the process group and return `False` so the orchestrator knows to exit. On success, it returns `True`.

Second, jobs occasionally crash right while writing the `latest` checkpoint pointer. This leaves a corrupted `latest` file that points to an incomplete checkpoint (e.g., missing `model.pt`), causing crash loops on restart. We need a function `get_safe_resume_checkpoint` that reads the `latest` pointer, validates that `model.pt` exists inside the target directory, and if it fails, falls back to scanning the folder for the highest complete `step_N` directory.

Finally, write a regression test in `tests/test_regression.py` that verifies `get_safe_resume_checkpoint` properly ignores a corrupted `latest` pointer and falls back to a valid step directory.
