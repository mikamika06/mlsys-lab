import sys

sys.path.insert(0, ".")
from keda.manifest import generate_scaled_object
from keda.validator import validate_scaled_object

CONFIG = {
    "name": "test-vllm",
    "namespace": "default",
    "service_name": "test-svc",
    "prometheus_server": "http://prom:9090",
    "target_num_requests": 4,
    "min_replicas": 1,
    "max_replicas": 5,
    "cooldown_period": 300,
    "polling_interval": 15
}


def test_scaled_object_structure():
    obj = generate_scaled_object(CONFIG)
    assert obj["kind"] == "ScaledObject"
    assert obj["apiVersion"] == "keda.sh/v1alpha1"


def test_min_max_replicas_invariant():
    obj = generate_scaled_object(CONFIG)
    spec = obj["spec"]
    assert spec["minReplicaCount"] <= spec["maxReplicaCount"]


def test_validator_accepts_valid_manifest():
    obj = generate_scaled_object(CONFIG)
    assert validate_scaled_object(obj) is True


def test_prometheus_trigger_present():
    obj = generate_scaled_object(CONFIG)
    triggers = obj["spec"]["triggers"]
    found = any(t.get("type") == "prometheus" for t in triggers)
    assert found
