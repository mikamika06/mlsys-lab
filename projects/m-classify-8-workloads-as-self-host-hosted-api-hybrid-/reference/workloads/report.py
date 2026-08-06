"""Reporting module."""
from workloads.classifier import classify_all

def generate_report(workloads):
    cls = classify_all(workloads)
    return {wid: info["deployment"] for wid, info in cls.items()}
