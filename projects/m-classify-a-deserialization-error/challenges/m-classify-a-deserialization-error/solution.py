import json
import pickle
try:
    import yaml
except ImportError:
    yaml = None

def classify_deserialization_error(err: Exception) -> str:
    if isinstance(err, json.JSONDecodeError):
        return "json_error"
    if isinstance(err, pickle.UnpicklingError):
        return "pickle_error"
    if yaml and isinstance(err, yaml.YAMLError):
        return "yaml_error"
    return "unknown_error"
