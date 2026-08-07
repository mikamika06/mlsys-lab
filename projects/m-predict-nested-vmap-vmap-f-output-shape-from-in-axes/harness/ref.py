import numpy as np

TEST_CASES = [
    {
        "input_shape": (10, 20),
        "in_axes_outer": 0,
        "in_axes_inner": 0,
        "base_out_shape": (5,),
        "batch_outer": 4,
        "batch_inner": 3,
    },
    {
        "input_shape": (30, 40, 50),
        "in_axes_outer": 1,
        "in_axes_inner": 2,
        "base_out_shape": (8, 2),
        "batch_outer": 5,
        "batch_inner": 6,
    },
    {
        "input_shape": (15,),
        "in_axes_outer": 0,
        "in_axes_inner": None,
        "base_out_shape": (3,),
        "batch_outer": 2,
        "batch_inner": 7,
    },
    {
        "input_shape": (8, 8),
        "in_axes_outer": None,
        "in_axes_inner": 0,
        "base_out_shape": (4,),
        "batch_outer": 3,
        "batch_inner": 4,
    },
    {
        "input_shape": (12, 16),
        "in_axes_outer": 1,
        "in_axes_inner": 1,
        "base_out_shape": (10,),
        "batch_outer": 2,
        "batch_inner": 2,
    },
]


def predict_nested_vmap_shape(input_shape, in_axes_outer, in_axes_inner, base_out_shape, batch_outer, batch_inner):
    return (batch_outer, batch_inner) + tuple(base_out_shape)


def simulate_shard_vs_pmap(arr, axis_name):
    return np.sum(arr, axis=0, keepdims=True)
