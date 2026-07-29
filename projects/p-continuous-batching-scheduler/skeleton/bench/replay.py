import json
import sys

sys.path.insert(0, ".")
from sched import Scheduler


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(trace_path, config_path):
    trace = load(trace_path)
    cfg = load(config_path)
    s = Scheduler(cfg)
    s.add(trace)
    m = s.run()
    for k in sorted(m):
        print(f"{k:20} {m[k]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
