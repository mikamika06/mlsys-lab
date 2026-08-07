class Server:
    """Inference server managing compilation cache."""
    def __init__(self, cache=None):
        from compcache.cache import CompilationCache
        self.cache = cache if cache is not None else CompilationCache()
        self.warmed = False

    def handle_first_request(self, req):
        self.warmed = True
        val = self.cache.lookup(req)
        if val is not None:
            return 1
        return 5

    def is_warmed(self):
        return self.warmed
