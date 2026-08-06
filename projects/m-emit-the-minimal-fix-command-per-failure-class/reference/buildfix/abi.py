def predict_abi_mismatch(torch_cxx11, host_cxx11):
    return torch_cxx11 != host_cxx11
