import numpy as np


class DeviceMesh2D:
    """Represents a 2D Device Mesh mapping global device IDs to multi-axis layouts."""

    def __init__(self, mesh_shape, mesh_dim_names, device_ids=None):

        if len(mesh_shape) != 2 or len(mesh_dim_names) != 2:
            raise ValueError("DeviceMesh2D requires exactly 2 dimensions")
        self.mesh_shape = tuple(mesh_shape)
        self.mesh_dim_names = tuple(mesh_dim_names)
        total_devices = self.mesh_shape[0] * self.mesh_shape[1]
        if device_ids is None:
            self.device_ids = np.arange(total_devices).reshape(self.mesh_shape)
        else:
            arr = np.array(device_ids)
            if arr.size != total_devices:
                raise ValueError("Device IDs size does not match mesh shape")
            self.device_ids = arr.reshape(self.mesh_shape)

    def get_rank_coords(self, global_rank):

        pos = np.where(self.device_ids == global_rank)
        if len(pos[0]) == 0:
            raise ValueError(f"Rank {global_rank} not found in mesh")
        return int(pos[0][0]), int(pos[1][0])

    def get_dim_group_ranks(self, global_rank, dim_name):

        if dim_name not in self.mesh_dim_names:
            raise ValueError(f"Unknown dimension {dim_name}")
        r, c = self.get_rank_coords(global_rank)
        if dim_name == self.mesh_dim_names[0]:
            return self.device_ids[:, c].tolist()
        else:
            return self.device_ids[r, :].tolist()

    def is_fast_axis(self, dim_name):

        if dim_name not in self.mesh_dim_names:
            raise ValueError(f"Unknown dimension {dim_name}")
        return dim_name == self.mesh_dim_names[1]

    def to_dict(self):

        return {
            "mesh_shape": list(self.mesh_shape),
            "mesh_dim_names": list(self.mesh_dim_names),
            "device_ids": self.device_ids.tolist(),
        }
