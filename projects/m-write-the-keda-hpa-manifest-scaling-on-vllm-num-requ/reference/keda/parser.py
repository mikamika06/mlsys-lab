def get_prometheus_query(config):
    return f'sum(vllm:num_requests_waiting{{service="{config["service_name"]}"}})'
