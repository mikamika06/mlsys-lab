class ProfileRuntimeEngine:
    """Runtime engine for profile context selection and safe enqueue."""

    def __init__(self, profiles):
        self.profiles = profiles

    def resolve_profile(self, input_shapes):
        """Find matching profile index for given input shapes."""
        for idx, prof in enumerate(self.profiles):
            matches = True
            for tensor_name, shape in input_shapes.items():
                if tensor_name not in prof:
                    matches = False
                    break
                bounds = prof[tensor_name]
                min_s = bounds["min"]
                max_s = bounds["max"]
                
                if len(shape) != len(min_s):
                    matches = False
                    break
                
                for dim_val, min_v, max_v in zip(shape, min_s, max_s):
                    if not (min_v <= dim_val <= max_v):
                        matches = False
                        break
                if not matches:
                    break
            if matches:
                return idx
        return -1

    def safe_enqueue(self, input_shapes):
        """Enqueue execution safely, selecting active profile or raising error if out-of-profile."""
        prof_idx = self.resolve_profile(input_shapes)
        if prof_idx < 0:
            raise ValueError(f"Out-of-profile input shapes: {input_shapes}")
        return {"status": "ok", "profile_index": prof_idx, "shapes": input_shapes}
