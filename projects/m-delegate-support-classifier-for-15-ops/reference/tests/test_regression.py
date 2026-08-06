from delpipeline.costmodel import compute_copy_cost

def test_partition_cost_regression():
    partitions = [{"boundary_tensors": ["t1"]}]
    tensor_metadata = {"t1": [10, 10]}
    cost = compute_copy_cost(partitions, tensor_metadata)
    assert cost > 0.0, "Cost must be positive"
    assert cost == 400.0, "Expected 400 bytes copy cost"
