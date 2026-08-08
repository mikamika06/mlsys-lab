import ref


def check(workdir):
    from exportcheck.classifier import classify_error

    out = {"classification_accuracy": 0.0}
    correct = 0
    total = len(ref.ERROR_LOG_TESTS)
    for log_text, expected in ref.ERROR_LOG_TESTS:
        try:
            res = classify_error(log_text)
            if res == expected:
                correct += 1
            elif "_note" not in out:
                out["_note"] = f"log '{log_text[:30]}': got '{res}', want '{expected}'"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"classifier raised {type(e).__name__}: {str(e)[:100]}"
    out["classification_accuracy"] = float(correct) / float(total) if total > 0 else 0.0
    return out
