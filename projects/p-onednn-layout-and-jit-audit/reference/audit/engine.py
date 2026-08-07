import json

class AuditEngine:
    def __init__(self, config):
        self.config = config
        self.logged = False
        self.events = [
            {"type": "exec", "layer": "conv1", "layout": "nchw"},
            {"type": "reorder", "from": "nchw", "to": "nhwc"},
            {"type": "exec", "layer": "relu1", "layout": "nhwc"},
            {"type": "reorder", "from": "nhwc", "to": "nchw"},
            {"type": "exec", "layer": "conv2", "layout": "nchw"}
        ]
        self.optimized = False

    def enable_log(self):
        self.logged = True
        return 1

    def parse_events(self):
        if not self.logged:
            return []
        return self.events

    def get_transitions(self):
        return [e for e in self.events if e["type"] == "reorder"]

    def find_redundant(self):
        reorders = self.get_transitions()
        if len(reorders) >= 2:
            return 1
        return 0

    def optimize_sequence(self):
        self.optimized = True
        self.events = [
            {"type": "exec", "layer": "conv1", "layout": "nchw"},
            {"type": "exec", "layer": "relu1", "layout": "nchw"},
            {"type": "exec", "layer": "conv2", "layout": "nchw"}
        ]
        return 1

    def run_inference(self):
        reorders = len(self.get_transitions())
        total = len(self.events)
        if total == 0:
            return 1.0
        return float(reorders) / float(total)
