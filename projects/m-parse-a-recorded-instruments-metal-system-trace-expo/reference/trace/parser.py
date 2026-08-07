"""Metal System Trace XML export parser."""

import xml.etree.ElementTree as ET


def parse_trace_events(xml_string):
    root = ET.fromstring(xml_string)
    events = []
    for event in root.findall(".//event"):
        events.append({
            "kind": event.get("kind", "compute"),
            "duration_us": float(event.get("duration_us", 0.0)),
            "label": event.get("label", ""),
        })
    return events


def count_command_buffers(events):
    counts = {"compute": 0, "blit": 0, "present": 0}
    for e in events:
        k = e.get("kind", "compute")
        if k in counts:
            counts[k] += 1
        else:
            counts[k] = 1
    return counts


def calculate_gpu_duty_cycle(events):
    if not events:
        return 0.0
    compute_time = sum(e["duration_us"] for e in events if e.get("kind") == "compute")
    total_time = sum(e["duration_us"] for e in events)
    if total_time == 0.0:
        return 0.0
    return compute_time / total_time
