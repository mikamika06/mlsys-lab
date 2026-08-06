import ref

def check(workdir):
    from extractor.decompile import verify_resume_signature
    code_snippet = "def resume_foo(a: torch.Tensor, b: int) -> torch.Tensor:\n    return a + b"
    expected_sig = "(a: torch.Tensor, b: int) -> torch.Tensor"

    try:
        res = verify_resume_signature(code_snippet, expected_sig)
    except Exception:
        return {"signature_matched": 0.0, "_note": "verification raised an exception"}

    if res:
        return {"signature_matched": 1.0}
    return {"signature_matched": 0.0, "_note": "signature did not match"}
