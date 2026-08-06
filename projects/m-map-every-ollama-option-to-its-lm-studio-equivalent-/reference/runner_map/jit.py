class ModelRunner:
    def __init__(self):
        self.loaded = False
        self.load_count = 0

    def request(self, payload):
        if not self.loaded:
            self.loaded = True
            self.load_count += 1
        return {"status": "success", "load_count": self.load_count}
