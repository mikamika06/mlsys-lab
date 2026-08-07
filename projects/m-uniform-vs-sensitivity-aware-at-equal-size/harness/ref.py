import random
import itertools

def uniform_alloc(profile, exclude, budget, base_bits=16):
    alloc = {}
    cost = 0
    active = []
    for p in profile:
        if p["name"] in exclude:
            alloc[p["name"]] = base_bits
            cost += p["params"] * base_bits
        else:
            active.append(p)

    if not active:
        if cost > budget:
            raise ValueError()
        return alloc

    common = set(active[0]["sens"].keys())
    for p in active[1:]:
        common &= set(p["sens"].keys())

    for b in sorted(common, reverse=True):
        cand_cost = cost + sum(p["params"] * b for p in active)
        if cand_cost <= budget:
            for p in active:
                alloc[p["name"]] = b
            return alloc
    raise ValueError()

def optimal_alloc(profile, exclude, budget, base_bits=16):
    alloc = {}
    cost = 0
    active = []
    for p in profile:
        if p["name"] in exclude:
            alloc[p["name"]] = base_bits
            cost += p["params"] * base_bits
        else:
            active.append(p)

    rem = budget - cost
    if rem < 0:
        raise ValueError()

    best_s = float('inf')
    best_c = None

    opts = [sorted(p["sens"].keys(), reverse=True) for p in active]
    for combo in itertools.product(*opts):
        b_cost = sum(active[i]["params"] * combo[i] for i in range(len(active)))
        if b_cost <= rem:
            s_cost = sum(active[i]["sens"][combo[i]] for i in range(len(active)))
            if s_cost < best_s:
                best_s = s_cost
                best_c = combo

    if best_c is None:
        raise ValueError()

    for i, p in enumerate(active):
        alloc[p["name"]] = best_c[i]
    return alloc

def greedy_alloc(profile, exclude, budget, base_bits=16):
    alloc = {}
    cost = 0
    active = []
    for p in profile:
        if p["name"] in exclude:
            alloc[p["name"]] = base_bits
            cost += p["params"] * base_bits
        else:
            bits = sorted(p["sens"].keys(), reverse=True)
            alloc[p["name"]] = bits[0]
            cost += p["params"] * bits[0]
            active.append({"name": p["name"], "params": p["params"], "sens": p["sens"], "bits": bits})

    while cost > budget:
        best_pen = float('inf')
        best_p = None
        for p in active:
            cur = alloc[p["name"]]
            idx = p["bits"].index(cur)
            if idx + 1 < len(p["bits"]):
                nxt = p["bits"][idx + 1]
                ds = p["sens"][nxt] - p["sens"][cur]
                db = (cur - nxt) * p["params"]
                pen = ds / db
                if pen < best_pen or (pen == best_pen and p["name"] < (best_p["name"] if best_p else "")):
                    best_pen = pen
                    best_p = p
        if not best_p:
            raise ValueError()

        cur = alloc[best_p["name"]]
        nxt = best_p["bits"][best_p["bits"].index(cur) + 1]
        alloc[best_p["name"]] = nxt
        cost -= (cur - nxt) * best_p["params"]

    return alloc

def eval_alloc(profile, exclude, alloc, base_bits=16):
    b = 0
    s = 0.0
    for p in profile:
        bit = alloc[p["name"]]
        b += p["params"] * bit
        if p["name"] not in exclude:
            s += p["sens"][bit]
    return b, s

def generate_profiles():
    random.seed(42)
    out = []
    for i in range(10):
        prof = []
        for j in range(4):
            prof.append({
                "name": f"L{j}",
                "params": random.randint(10, 50) * 10,
                "sens": {8: random.uniform(0.1, 0.5), 4: random.uniform(1.0, 3.0), 2: random.uniform(5.0, 15.0)}
            })
        ex = ["L0"] if i % 2 == 0 else []
        min_bits = sum(p["params"] * 2 for p in prof if p["name"] not in ex) + sum(p["params"] * 16 for p in prof if p["name"] in ex)
        max_bits = sum(p["params"] * 8 for p in prof if p["name"] not in ex) + sum(p["params"] * 16 for p in prof if p["name"] in ex)
        bdg = random.randint(min_bits + 10, max_bits - 10)
        out.append((prof, ex, bdg))
    return out

PROFILES = generate_profiles()
