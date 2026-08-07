def verify_adapter_shape(in_features, out_features, rank):
    return [(out_features, rank), (rank, in_features)]
