from keda.parser import get_prometheus_query


def generate_scaled_object(config):
    q = get_prometheus_query(config)
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
