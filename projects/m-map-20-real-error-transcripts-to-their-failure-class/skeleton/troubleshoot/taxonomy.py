"""Failure taxonomy classification and diagnosis."""


def classify_transcript(transcript: str, ps_output: str, mem_counters: dict) -> str:
    raise NotImplementedError


def diagnose_failure(transcript: str, ps_output: str, mem_counters: dict) -> dict:
    raise NotImplementedError
