from typing import Any, Dict


def build_naive_graph_spec(embed_dim: int, num_heads: int, seq_len: int,
    qkv_w: Any, out_w: Any) ->Dict[str, Any]:
    raise NotImplementedError


def build_ane_graph_spec(embed_dim: int, num_heads: int, seq_len: int,
    qkv_w: Any, out_w: Any) ->Dict[str, Any]:
    raise NotImplementedError


def count_layout_ops(graph_spec: Dict[str, Any]) ->int:
    raise NotImplementedError


def simulate_execution(graph_spec: Dict[str, Any], target: str) ->float:
    raise NotImplementedError
