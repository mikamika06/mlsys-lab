We are running performance profiling sweeps inside a containerized ML serving and profiling cluster using NVIDIA Nsight Systems (nsys) and Nsight Compute (ncu). Recently, one of our automated profiling nodes started throwing opaque errors whenever a profiling job kicks off for custom fused attention kernels. Instead of yielding the expected kernel metrics or timeline traces, the profiling run aborts immediately or reports cryptic permission errors related to performance counters, system profiling capabilities, or perf events.

Interestingly, running the exact same workload directly on a bare-metal machine with administrator/root privileges succeeds without issues, but inside our standard container environment, it fails. We suspect this is either a missing container capability or an underlying host-level permission or kernel configuration failure (such as restrictive `perf_event_paranoid` settings or missing `SYS_ADMIN`).

Your task is to build a diagnostic tool in `profilediag/` that:
1. Disambiguates whether a failed profiling run is caused by a missing container capability `--cap-add=SYS_ADMIN` versus a general bare-metal or system-level permission failure (e.g., `kernel.perf_event_paranoid` restrictions, driver mismatch, or missing device nodes).
2. Categorizes and triages 6 real failure transcripts obtained from `ncu` and `nsys` logs into their exact root-cause categories.
3. Provides a robust regression test suite ensuring that the triage engine correctly identifies failure modes and does not falsely classify container privilege issues as hardware or driver faults.

Read the specifications and implement the required modules under `profilediag/` and the tests under `tests/test_regression.py`.
