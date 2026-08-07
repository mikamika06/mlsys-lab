import ref
import torch


def check(workdir):
    from optmem.measure import compare_full_vs_lora

    m_full = ref.DummyModel(128, False)
    m_lora = ref.DummyModel(128, True)

    out = {"ratio_match": 0.0}

    try:
        got = compare_full_vs_lora(m_full, m_lora)
    except NotImplementedError:
        return out

    want_full = ref.step_and_count(ref.DummyModel(128, False), torch.optim.AdamW)
    want_lora = ref.step_and_count(ref.DummyModel(128, True), torch.optim.AdamW)
    want_ratio = want_lora / want_full if want_full else 0.0

    if isinstance(got, dict) and "size_ratio" in got:
        if abs(got["size_ratio"] - want_ratio) < 1e-4:
            out["ratio_match"] = 1.0
        else:
            out["_note"] = f"Expected size_ratio approx {want_ratio:.4f}, got {got.get('size_ratio')}"
    else:
        out["_note"] = f"Missing size_ratio in return dict: {got}"

    return out
