def get_oracle_dataset():
    return {
        0.1: "context_start",
        0.5: "context_middle",
        0.9: "context_end"
    }

def mock_model_good(text):
    return "Found SECRET_FACT" if "SECRET_FACT" in text else "Lost"

def mock_model_bad(text):
    if "middle" in text or 0.4 < text.get("pos", 0.5) < 0.6:
        return "Lost"
    return "Found SECRET_FACT"
