"""mlsys-lab — the grading engine.

A task ships a solver contract, a reference implementation and a set of gates.
The engine runs the learner's code locally, computes a deterministic NUMBER, and
checks it against those gates. Every metric is hardware-independent (miss counts
from a modelled cache, memory transactions from a simulated GPU, size ratios, KL,
byte-exact fractions); wall-clock time is never a gate, so the same submission
scores identically on any machine.

Three languages, three runners:
  * Python  — the solver is imported and called directly (`mlsys.runner`)
  * C++     — `solve.cpp` is compiled with the local clang++ and executed
              (`mlsys.runners.cpp`)
  * CUDA-C  — `solve.cu` is parsed and executed thread-by-thread on the software
              GPU in `mlsys.sim` (`mlsys.runners.cuda`)

Task-facing API (what a task's `check.py` imports):

    from mlsys import scorers, probe
    from mlsys import cachesim, cppabi   # the deterministic cache and ABI models
    from mlsys.sim import GPU, CudaProgram
"""

from . import probe, scorers
from .sim import abi as cppabi          # historical names a task may still use
from .sim import cache as cachesim

__version__ = "0.3.0"
__all__ = ["scorers", "probe", "cachesim", "cppabi", "__version__"]
