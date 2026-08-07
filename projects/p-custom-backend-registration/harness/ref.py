from compiler.backend import BackendRegistry, CompilationGraph
from compiler.passes import CustomOptimizationPass, check_equivalence, apply_with_fallback

class DummyBackend:
    pass
