import numpy as np
from edge_model.state import define_state_contract
from edge_model.export import convert_stateful_model
from edge_model.runtime import StatefulRunner

def get_oracle_contract():
    return define_state_contract(2, 4, 16, 128)

def get_oracle_export():
    c = get_oracle_contract()
    return convert_stateful_model(c)

def run_cached_vs_uncached(tokens):
    contract = get_oracle_contract()
    runner = StatefulRunner(contract)
    outputs = []
    for t in tokens:
        out, _ = runner.step(t)
        outputs.append(out[0])
    return outputs
