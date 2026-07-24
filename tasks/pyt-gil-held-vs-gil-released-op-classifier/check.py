import hashlib
import socket
import threading
import time

import numpy as np


def _build_ops():
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

    return {
        "sleep": sleep_op,
        "dot": dot_op,
        "hash": hash_op,
        "python_loop": python_loop_op,
        "socket_recv": socket_recv_op,
    }


def _gil_oracle(fn):
    counter = {"value": 0}
    running = {"value": True}

    def worker():
        while running["value"]:
            counter["value"] += 1

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.02)
    before = counter["value"]
    fn()
    after = counter["value"]
    running["value"] = False
    thread.join()

    return (after - before) > 1000


def grade(sol, fx) -> dict:
    ops = _build_ops()
    expected = {name: _gil_oracle(fn) for name, fn in ops.items()}
    try:
        got = sol.classify_gil_release(list(ops.keys()))
    except Exception:
        return {"exact_match": 0.0}

    if not isinstance(got, dict):
        return {"exact_match": 0.0}

    return {"exact_match": 1.0 if got == expected else 0.0}
