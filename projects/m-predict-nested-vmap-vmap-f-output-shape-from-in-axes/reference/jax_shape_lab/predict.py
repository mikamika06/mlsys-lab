def predict_nested_vmap_shape(input_shape, in_axes_outer, in_axes_inner, base_out_shape, batch_outer, batch_inner):
    return (batch_outer, batch_inner) + tuple(base_out_shape)
