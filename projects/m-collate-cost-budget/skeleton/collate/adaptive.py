"""Adaptive collate batching within strict budgets."""

class AdaptiveCollatePlanner:
    """Calculates maximum compliant batch size and optional feature flags for collate budget."""

    def __init__(self, collate_fn, sample_generator, max_budget_ms):
        raise NotImplementedError

    def compute_max_safe_batch_size(self, max_batch_size):
        """Find the largest batch size <= max_batch_size that stays within max_budget_ms."""
        raise NotImplementedError

    def optimize_features_and_batch_size(self, candidate_feature_sets, target_batch_size):
        """Find the optimal combination of features and batch size within budget."""
        raise NotImplementedError
