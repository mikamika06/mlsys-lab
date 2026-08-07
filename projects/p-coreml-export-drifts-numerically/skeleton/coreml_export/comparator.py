class LayerComparator:
    def compare_layers(self):
        raise NotImplementedError

    def has_max_diff(self):
        raise NotImplementedError

    def find_top_culprit(self):
        raise NotImplementedError
