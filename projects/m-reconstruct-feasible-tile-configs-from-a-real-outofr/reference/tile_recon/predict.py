def predict_out_of_resources(max_smem, element_size, block_m, block_n, stages, overhead=0):
    smem = (block_m * block_n * element_size * stages) + overhead
    return smem > max_smem
