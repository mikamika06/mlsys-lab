import hashlib
import socket
import threading
import time

import numpy as np


def _probe(fn):
    counter = {"value": 0}
    running = {"value": True}

    def worker():
        while running["value"]:
            counter["value"] += 1

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.02)
    before = counter["value"]
    fn()
    after = counter["value"]
    running["value"] = False
    t.join()
    return (after - before) > 1000


def classify_gil_release(ops):
    def sleep_op():
        time.sleep(0.15)

    def dot_op():
        a = np.ones((700, 700), dtype=np.float64)
        np.dot(a, a)

    def hash_op():
        data = b"x" * (32 * 1024 * 1024)
        hashlib.sha256(data).digest()

    def python_loop_op():
        total = 0
        for i in range(6000000):
            total += i
        return total

    def socket_recv_op():
        left, right = socket.socketpair()

        def sender():
            time.sleep(0.15)
            right.send(b"x")

        t = threading.Thread(target=sender)
        t.start()
        left.recv(1)
        t.join()
        left.close()
        right.close()

    available = {
        "sleep": sleep_op,
        "dot": dot_op,
        "hash": hash_op,
        "python_loop": python_loop_op,
        "socket_recv": socket_recv_op,
    }

    return {name: _probe(available[name]) for name in ops}
