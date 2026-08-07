from preempt.model import recompute_cost, swap_cost, breakeven_length
from preempt.policy import PreemptionPolicy
from preempt.scheduler import PreemptionScheduler

def get_oracle_recompute(ctx, hs, nl, tf):
    return recompute_cost(ctx, hs, nl, tf)

def get_oracle_swap(tokens, bpt, bw):
    return swap_cost(tokens, bpt, bw)

def get_oracle_breakeven(hs, nl, tf, bpt, bw):
    return breakeven_length(hs, nl, tf, bpt, bw)
