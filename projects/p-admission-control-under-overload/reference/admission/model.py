class Request:
    def __init__(self, req_id, priority, cost, arrival_time):
        self.id = req_id
        self.priority = priority
        self.cost = cost
        self.arrival_time = arrival_time

class QueueModel:
    def __init__(self, capacity=100, processing_rate=10.0):
        self.capacity = capacity
        self.processing_rate = processing_rate
        self.queue = []

    def enqueue(self, req):
        if len(self.queue) >= self.capacity:
            return False
        self.queue.append(req)
        return True

    def estimated_wait_time(self):
        total_cost = sum(r.cost for r in self.queue)
        return total_cost / self.processing_rate

    def clear(self):
        self.queue.clear()
