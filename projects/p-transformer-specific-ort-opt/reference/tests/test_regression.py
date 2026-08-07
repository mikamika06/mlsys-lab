import sys
sys.path.insert(0, ".")
from opt.transformer_fusion import apply_transformer_fusion
from opt.analyzer import analyze_fused_nodes
from opt.validator import check_parity

def test_fusion_applied():
    g = {"nodes": [{"name": "att", "op": "AttentionSubGraph", "inputs": ["q", "k", "v"], "outputs": ["out"]}]}
    res = apply_transformer_fusion(g)
    assert res["fused_count"] >= 1

def test_parity_check():
    g_orig = {}
    g_opt = {}
    res = check_parity(g_orig, g_opt, {})
    assert res["parity_ok"] == 1
