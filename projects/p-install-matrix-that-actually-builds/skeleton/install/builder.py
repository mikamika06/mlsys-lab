def lock_dependencies():
    raise NotImplementedError

def build_on_config(config_id):
    raise NotImplementedError

def verify_fresh_install():
    raise NotImplementedError

def execute_fallback(hardware_spec):
    raise NotImplementedError
