class CandidateTree:
    def __init__(self, roots):
        raise NotImplementedError

    def add_child(self, parent_id, token_id, prob):
        raise NotImplementedError

    def get_paths(self):
        raise NotImplementedError


def build_tree_from_logits(draft_logits, max_depth, max_width):
    raise NotImplementedError
