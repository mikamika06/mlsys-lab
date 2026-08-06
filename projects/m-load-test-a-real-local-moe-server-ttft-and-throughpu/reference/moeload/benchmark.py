def run_benchmark_session(server, workload_requests, concurrency):
    traces = []
    for req in workload_requests:
        prompt_toks = req["prompt_tokens"]
        decode_toks = req["decode_tokens"]
        trace = server.process_request(prompt_toks, decode_toks, concurrency=concurrency)
        traces.append(trace)
    return traces
