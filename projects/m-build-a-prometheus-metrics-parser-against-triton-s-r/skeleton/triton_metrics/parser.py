from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricSample:
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: Optional[int] = None


def parse_prometheus_text(text: str) -> List[MetricSample]:
    """Parse raw Prometheus text into a list of MetricSample objects."""
    raise NotImplementedError
