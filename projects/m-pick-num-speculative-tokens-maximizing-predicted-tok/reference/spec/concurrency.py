import ref

def decide_speculation(concurrency, acceptance_rate, model):
    return ref.decide_go_no_go(concurrency, acceptance_rate, model)
