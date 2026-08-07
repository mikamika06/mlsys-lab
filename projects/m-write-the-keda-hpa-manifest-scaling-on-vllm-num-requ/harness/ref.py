CONFIGS = [
    {
        "name": "vllm-deployment-a",
        "namespace": "default",
        "service_name": "vllm-service",
        "prometheus_server": "http://prometheus-k8s.monitoring.svc.cluster.local:9090",
        "target_num_requests": 5,
        "min_replicas": 1,
        "max_replicas": 10,
        "cooldown_period": 300,
        "polling_interval": 15
    },
    {
        "name": "vllm-deployment-b",
        "namespace": "llm-inference",
        "service_name": "vllm-inference-svc",
        "prometheus_server": "http://thanos-query.monitoring.svc.cluster.local:9090",
        "target_num_requests": 8,
        "min_replicas": 2,
        "max_replicas": 20,
        "cooldown_period": 600,
        "polling_interval": 10
    },
    {
        "name": "vllm-deployment-c",
        "namespace": "production",
        "service_name": "llama-service",
        "prometheus_server": "http://prom.prod.svc:9090",
        "target_num_requests": 2,
        "min_replicas": 1,
        "max_replicas": 5,
        "cooldown_period": 120,
        "polling_interval": 5
    }
]

def build_query(config):
    name = config["name"]
    return f'sum(vllm:num_requests_waiting{{service="{config["service_name"]}"}})'

def build_manifest(config):
    q = build_query(config)
    return {
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {
            "name": f"{config['name']}-scaler",
            "namespace": config["namespace"]
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": config["name"]
            },
            "minReplicaCount": config["min_replicas"],
            "maxReplicaCount": config["max_replicas"],
            "cooldownPeriod": config["cooldown_period"],
            "pollingInterval": config["polling_interval"],
            "triggers": [
                {
                    "type": "prometheus",
                    "metadata": {
                        "serverAddress": config["prometheus_server"],
                        "metricName": "vllm_num_requests_waiting",
                        "query": q,
                        "threshold": str(config["target_num_requests"])
                    }
                }
            ]
        }
    }
