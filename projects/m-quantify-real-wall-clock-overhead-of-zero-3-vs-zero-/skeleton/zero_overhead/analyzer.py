def parse_step_logs(logs):
    """Parse step timing dictionaries for ZeRO runs."""
    raise NotImplementedError


def extract_stage_summary(parsed_logs, stage):
    """Extract mean timing components for a given ZeRO stage."""
    raise NotImplementedError
