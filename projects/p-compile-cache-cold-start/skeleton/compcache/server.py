class Server:
    def __init__(self, cache=None):
        raise NotImplementedError

    def handle_first_request(self, req):
        raise NotImplementedError

    def is_warmed(self):
        raise NotImplementedError
