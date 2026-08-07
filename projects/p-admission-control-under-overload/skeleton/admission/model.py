class Request:
    def __init__(self, req_id, priority, cost, arrival_time):
        raise NotImplementedError

class QueueModel:
    def __init__(self, capacity=100, processing_rate=10.0):
        raise NotImplementedError

    def enqueue(self, req):
        raise NotImplementedError

    def estimated_wait_time(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
