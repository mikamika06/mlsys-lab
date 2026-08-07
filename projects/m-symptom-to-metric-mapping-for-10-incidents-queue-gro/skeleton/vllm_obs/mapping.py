"""Symptom-to-metric mapping for vLLM cluster incidents."""


def map_incident_to_metric(incident_id: int) -> dict:
    raise NotImplementedError


def parse_telemetry_sample(metrics: dict) -> dict:
    raise NotImplementedError
