from typing import Any, Dict


def build_naive_graph_spec(embed_dim: int, num_heads: int, seq_len: int,
    qkv_w: Any, out_w: Any) ->Dict[str, Any]:
    return {
        'format': 'mlpackage',
        'layout': 'BSHD',
        'ops': [
            {'name': 'qkv_proj', 'type': 'matmul'},
            {'name': 'q_reshape', 'type': 'reshape'},
            {'name': 'q_transpose', 'type': 'transpose'},
            {'name': 'k_reshape', 'type': 'reshape'},
            {'name': 'k_transpose', 'type': 'transpose'},
            {'name': 'v_reshape', 'type': 'reshape'},
            {'name': 'v_transpose', 'type': 'transpose'},
            {'name': 'attn_matmul', 'type': 'matmul'},
            {'name': 'softmax', 'type': 'softmax'},
            {'name': 'ctx_matmul', 'type': 'matmul'},
            {'name': 'out_transpose', 'type': 'transpose'},
            {'name': 'out_reshape', 'type': 'reshape'},
            {'name': 'out_proj', 'type': 'matmul'}
        ]
    }


def build_ane_graph_spec(embed_dim: int, num_heads: int, seq_len: int,
    qkv_w: Any, out_w: Any) ->Dict[str, Any]:
    return {
        'format': 'mlpackage',
        'layout': 'BC1S',
        'ops': [
            {'name': 'qkv_conv1x1', 'type': 'conv2d'},
            {'name': 'split_qkv', 'type': 'split'},
            {'name': 'batch_matmul_qk', 'type': 'matmul'},
            {'name': 'softmax', 'type': 'softmax'},
            {'name': 'batch_matmul_v', 'type': 'matmul'},
            {'name': 'out_conv1x1', 'type': 'conv2d'}
        ]
    }


def count_layout_ops(graph_spec: Dict[str, Any]) ->int:
    count = 0
    for op in graph_spec.get('ops', []):
        if op.get('type') in ('reshape', 'transpose'):
            count += 1
    return count


def simulate_execution(graph_spec: Dict[str, Any], target: str) ->float:
    ops = graph_spec.get('ops', [])
    layout_op_count = count_layout_ops(graph_spec)
    compute_ops = len(ops) - layout_op_count
    if target == 'CPU_AND_NE':
        if graph_spec.get('layout') == 'BC1S':
            return float(compute_ops * 1.0 + layout_op_count * 3.0)
        return float(compute_ops * 1.5 + layout_op_count * 8.0)
    return float(len(ops) * 2.0)
