def parse_structure(checkpoint_dir):
    raise NotImplementedError

def map_sharding(state_dict, world_size):
    raise NotImplementedError

def convert_to_portable(checkpoint_dir, output_path):
    raise NotImplementedError

def restore_from_portable(portable_path, target_world_size, rank):
    raise NotImplementedError
