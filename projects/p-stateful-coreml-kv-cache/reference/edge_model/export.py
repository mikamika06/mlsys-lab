import numpy as np

def convert_stateful_model(contract):
    package = {
        "compiled": True,
        "contract": contract,
        "weights_hash": 1337
    }
    return package
