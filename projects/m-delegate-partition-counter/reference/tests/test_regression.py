from delegate_partition.counter import count_partitions

def test_partition_counting_basic():
    graph = {"ops": ["Conv2D", "CustomOp", "Conv2D"], "supported": ["Conv2D"]}
    assert count_partitions(graph) == 2

def test_partition_counting_contiguous():
    graph = {"ops": ["Conv2D", "Conv2D", "Conv2D"], "supported": ["Conv2D"]}
    assert count_partitions(graph) == 1

def test_partition_counting_empty():
    graph = {"ops": ["CustomOp", "CustomOp"], "supported": ["Conv2D"]}
    assert count_partitions(graph) == 0
