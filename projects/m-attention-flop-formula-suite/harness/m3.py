import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_flawed_counter": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out
    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        return out
    if first is None:
        out["_note"] = "no test_* functions found"
        return out
    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import flopcount.transformer as t

    good_layer = t.count_layer_flops
    good_trans = t.count_transformer_flops

    def flawed_layer(
        seq_len,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        ffn_hidden_dim,
        causal=True,
        pass_type="fwd",
    ):
        q_proj = 2 * seq_len * hidden_dim * (num_heads * head_dim)
        k_proj = 2 * seq_len * hidden_dim * (num_kv_heads * head_dim)
        v_proj = 2 * seq_len * hidden_dim * (num_kv_heads * head_dim)
        out_proj = 2 * seq_len * (num_heads * head_dim) * hidden_dim
        attn_core = t.count_attention_flops(
            1, num_heads, num_kv_heads, seq_len, seq_len, head_dim, causal
        )
        gate_proj = 2 * seq_len * hidden_dim * ffn_hidden_dim
        down_proj = 2 * seq_len * ffn_hidden_dim * hidden_dim
        fwd_flops = (
            q_proj
            + k_proj
            + v_proj
            + out_proj
            + attn_core
            + gate_proj
            + down_proj
        )
        mult = 1 if pass_type == "fwd" else 2
        return mult * fwd_flops

    def flawed_transformer(
        num_layers,
        seq_len,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        ffn_hidden_dim,
        vocab_size,
        causal=True,
        pass_type="fwd",
    ):
        layer_f = flawed_layer(
            seq_len,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_hidden_dim,
            causal,
            pass_type,
        )
        return num_layers * layer_f

    t.count_layer_flops = flawed_layer
    t.count_transformer_flops = flawed_transformer
    import flopcount

    flopcount.count_layer_flops = flawed_layer
    flopcount.count_transformer_flops = flawed_transformer

    try:
        out["catches_flawed_counter"] = 0.0 if _survives(path) else 1.0
    finally:
        t.count_layer_flops = good_layer
        t.count_transformer_flops = good_trans
        flopcount.count_layer_flops = good_layer
        flopcount.count_transformer_flops = good_trans

    return out
