"""Verify communication-free sharded state dict placements."""


def verify_communication_free_state_dict(param_specs, world_size, rank):
    """Check if all parameter shards on this rank are local and communication-free."""
    raise NotImplementedError
