class ModelTracker:
    def __init__(self):
        self.states = {}

    def touch(self, model_name, time_step):
        self.states[model_name] = time_step
