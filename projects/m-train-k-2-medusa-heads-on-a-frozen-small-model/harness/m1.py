import ref

def check(workdir):
    from medusa.train import train_medusa_heads
    hs, targets = ref.generate_data()
    got = train_medusa_heads(hs, targets)
    want = ref.train_heads(hs, targets)
    rel = abs(got - want) / (abs(want) + 1e-8)
    return {"rel_err": float(rel)}
