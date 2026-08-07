import sys
sys.path.insert(0, ".")
from fa_fix import dispatcher
import ref

def test_strict_attention_raises_on_fallback():
    q = ref.create_tensor((1, 4, 64, 32), dtype="float32", aligned=False)
    k = ref.create_tensor((1, 4, 64, 32), dtype="float32", aligned=False)
    v = ref.create_tensor((1, 4, 64, 32), dtype="float32", aligned=False)
    try:
        dispatcher.strict_attention(q, k, v, strict=True)
        assert False, "Expected RuntimeError on silent fallback to math backend"
    except RuntimeError:
        pass

def test_fixed_inputs_use_fast_path():
    q = ref.create_tensor((1, 4, 64, 32), dtype="float32", aligned=False)
    q_f, k_f, v_f, _ = dispatcher.fix_inputs(q, q, q)
    b = dispatcher.get_backend(q_f, k_f, v_f)
    assert b == "flash_attention"
