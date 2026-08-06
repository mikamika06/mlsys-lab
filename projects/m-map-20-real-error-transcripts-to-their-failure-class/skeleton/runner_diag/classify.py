def classify_transcript(transcript: dict) -> dict:
    """Analyze error log, ps list, and memory state to identify failure."""
    raise NotImplementedError


def classify_all(transcripts: list[dict]) -> list[dict]:
    """Classify a list of error transcripts."""
    raise NotImplementedError
