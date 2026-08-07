import numpy as np
from streammetrics.parser import parse_stream_metrics


def compute_median_decode_throughput(runs_ndjson_streams, warmup_runs=1):
    valid_runs = runs_ndjson_streams[warmup_runs:]
    if not valid_runs:
        return 0.0
    throughputs = []
    for stream in valid_runs:
        metrics = parse_stream_metrics(stream)
        throughputs.append(metrics["decode_tok_per_sec"])
    return float(np.median(throughputs))
