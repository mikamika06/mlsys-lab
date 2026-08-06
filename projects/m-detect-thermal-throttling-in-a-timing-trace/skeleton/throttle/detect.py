"""Throttling detection routines."""


def extract_event_spans(is_throttled, min_consecutive=3):
    raise NotImplementedError


def detect_throttling_events(durations, window_size=5, threshold_factor=1.35, min_consecutive=3):
    raise NotImplementedError
