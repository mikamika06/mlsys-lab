import ref
import torch


def check(workdir):
    from optmem.qlora import measure_8bit_delta

    model = ref.DummyModel(128, False)
    out = {"delta_match": 0.0}

    try:
        got = measure_8bit_delta(model, torch.optim.AdamW, ref.AdamW8bitMock)
    except NotImplementedError:
        return out

    want_32 = ref.step_and_count(ref.DummyModel(128, False), torch.optim.AdamW)
    want_8 = ref.step_and_count(ref.DummyModel(128, False), ref.AdamW8bitMock)
    want_delta = want_32 - want_8

    if isinstance(got, dict) and "delta_bytes" in got:
        if got["delta_bytes"] == want_delta:
            out["delta_match"] = 1.0
        else:
            out["_note"] = f"Expected delta_bytes {want_delta}, got {got.get('delta_bytes')}"
    else:
        out["_note"] = f"Missing delta_bytes in return dict: {got}"

    return out
