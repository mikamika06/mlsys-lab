def detect_thrash(logs: list[dict], current_args: dict) -> dict | None:
    raise NotImplementedError


def classify_traces(traces: list[dict]) -> dict[str, str]:
    raise NotImplementedError
