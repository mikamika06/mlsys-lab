import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricSample:
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: Optional[int] = None


def parse_labels(raw_labels: str) -> Dict[str, str]:
    labels = {}
    if not raw_labels:
        return labels
    pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:\\.|[^\\"])*)"')
    for match in pattern.finditer(raw_labels):
        k, v = match.group(1), match.group(2)
        v = v.replace(r'\"', '"').replace(r'\n', '\n').replace(r'\\', '\\')
        labels[k] = v
    return labels


def parse_prometheus_text(text: str) -> List[MetricSample]:
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+(\d+))?$', line)
        if not match:
            continue

        name, raw_labels, val_str, ts_str = match.groups()
        labels = parse_labels(raw_labels) if raw_labels else {}

        try:
            val = float(val_str)
        except ValueError:
            continue

        ts = int(ts_str) if ts_str else None
        samples.append(MetricSample(name=name, labels=labels, value=val, timestamp=ts))

    return samples
