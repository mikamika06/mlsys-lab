import torch
import torch.fx

SYNC_TARGETS = {
    "item", "nonzero", "item_", "synchronize", "cudaSynchronize",
    "torch.cuda.synchronize"
}

MALLOC_TARGETS = {
    torch.empty, torch.zeros, torch.ones, torch.randn, torch.rand,
    "empty", "zeros", "ones", "randn", "rand"
}

def check_graph_violations(gm: torch.fx.GraphModule) -> list[dict]:
    """Detect CUDA Graph capture violations in FX GraphModule."""
    violations = []
    for node in gm.graph.nodes:
        if node.op == "call_method":
            if node.target in ("item", "nonzero"):
                violations.append({"node": node.name, "rule": "SYNC_OP", "severity": "HIGH"})
            elif node.target in ("cpu", "cuda", "to"):
                violations.append({"node": node.name, "rule": "H2D_TRANSFER", "severity": "MEDIUM"})
            elif node.target in ("empty", "zeros", "ones", "randn"):
                violations.append({"node": node.name, "rule": "MALLOC_OP", "severity": "HIGH"})
        elif node.op == "call_function":
            target_name = getattr(node.target, "__name__", str(node.target))
            if node.target in (torch.cuda.synchronize, torch.nonzero) or target_name in SYNC_TARGETS:
                violations.append({"node": node.name, "rule": "SYNC_OP", "severity": "HIGH"})
            elif node.target in (torch.empty, torch.zeros, torch.ones, torch.randn, torch.rand) or target_name in MALLOC_TARGETS:
                violations.append({"node": node.name, "rule": "MALLOC_OP", "severity": "HIGH"})
    return violations
