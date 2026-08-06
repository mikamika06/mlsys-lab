"""Adaptive collate batching within strict budgets."""

from collate.budget import profile_collate


class AdaptiveCollatePlanner:
    """Calculates maximum compliant batch size and optional feature flags for collate budget."""

    def __init__(self, collate_fn, sample_generator, max_budget_ms):
        self.collate_fn = collate_fn
        self.sample_generator = sample_generator
        self.max_budget_ms = max_budget_ms

    def compute_max_safe_batch_size(self, max_batch_size):
        """Find the largest batch size <= max_batch_size that stays within max_budget_ms."""
        best_bs = 0
        candidate_sizes = list(range(1, max_batch_size + 1))
        profile = profile_collate(self.collate_fn, self.sample_generator, candidate_sizes)

        for bs in candidate_sizes:
            if profile[bs]["total_time_ms"] <= self.max_budget_ms:
                best_bs = bs
        return best_bs

    def optimize_features_and_batch_size(self, candidate_feature_sets, target_batch_size):
        """Find the optimal combination of features and batch size within budget."""
        best_plan = None

        for feature_config in candidate_feature_sets:
            name = feature_config["name"]
            fn = feature_config["fn"]

            profile = profile_collate(fn, self.sample_generator, [target_batch_size])
            cost = profile[target_batch_size]["total_time_ms"]

            if cost <= self.max_budget_ms:
                if best_plan is None or feature_config.get("priority", 0) > best_plan.get("priority", -1):
                    best_plan = {
                        "name": name,
                        "batch_size": target_batch_size,
                        "cost_ms": cost,
                        "priority": feature_config.get("priority", 0)
                    }

        return best_plan
