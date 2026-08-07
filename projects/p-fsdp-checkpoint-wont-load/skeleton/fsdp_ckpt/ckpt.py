class FSDPCheckpointManager:
    def __init__(self, checkpoint_data):
        raise NotImplementedError

    def parse_structure(self):
        raise NotImplementedError

    def map_shards(self):
        raise NotImplementedError

    def convert_to_unified(self):
        raise NotImplementedError

    def load_on_cards(self, target_world_size):
        raise NotImplementedError

    def verify_loss(self, target_world_size, test_input):
        raise NotImplementedError
