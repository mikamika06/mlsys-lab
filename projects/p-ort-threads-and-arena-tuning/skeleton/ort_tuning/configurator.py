class RuntimeConfigurator:
    def __init__(self, hardware_info):
        raise NotImplementedError

    def measure_threads(self, thread_counts):
        raise NotImplementedError

    def configure_arena(self, enable=True):
        raise NotImplementedError

    def use_io_binding(self, enable=True):
        raise NotImplementedError

    def set_graph_optimization(self, level):
        raise NotImplementedError

    def get_latency(self):
        raise NotImplementedError

    def build_config(self):
        raise NotImplementedError
