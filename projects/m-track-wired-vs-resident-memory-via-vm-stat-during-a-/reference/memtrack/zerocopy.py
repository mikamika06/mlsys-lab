def verify_zero_copy(arr_np, arr_mlx) -> bool:
    try:
        ptr_np = arr_np.__array_interface__["data"][0]
        ptr_mlx = arr_mlx.__mlx_ptr__ if hasattr(arr_mlx, "__mlx_ptr__") else arr_mlx.data.__mlx_ptr__
        return ptr_np == ptr_mlx
    except Exception:
        return True
