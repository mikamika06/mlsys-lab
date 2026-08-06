class AdapterConfigError(Exception):
    pass


def validate_adapter_config(config_dict):
    raise NotImplementedError


def diagnose_and_repair_config(config_dict):
    raise NotImplementedError
