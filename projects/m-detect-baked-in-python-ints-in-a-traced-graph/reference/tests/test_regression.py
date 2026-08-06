from graphguard.classifier import classify_failures

def test_classifier_coverage():
    failures = [
        "Guard failed: tensor value is static int 5 instead of dynamic",
        "Found baked-in python integer constant in graph attribute",
        "Constant scalar literal embedded directly into node payload"
    ]
    results = classify_failures(failures)
    assert all(r == "baked_int" for r in results)
