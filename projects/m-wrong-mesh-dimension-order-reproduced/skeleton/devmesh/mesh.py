import numpy as np


class DeviceMesh2D:
    def __init__(self, mesh_shape, mesh_dim_names, device_ids=None):
        raise NotImplementedError

    def get_rank_coords(self, global_rank):
        raise NotImplementedError

    def get_dim_group_ranks(self, global_rank, dim_name):
        raise NotImplementedError

    def is_fast_axis(self, dim_name):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError
