from trtprof.profile import calculate_profile_bounds


def build_profile_plan(workload_spec):
    """Build TRT profile configurations for each named input tensor in workload spec."""
    profiles = []
    
    for prof_spec in workload_spec.get("profiles", []):
        prof_dict = {}
        strategy = prof_spec.get("strategy", "p50")
        padding = prof_spec.get("padding_ratio", 0.1)
        
        for tensor_name, data in prof_spec.get("tensors", {}).items():
            fixed_dims = data.get("fixed_dims", [])
            dynamic_samples = data.get("samples", [])
            
            min_d, opt_d, max_d = calculate_profile_bounds(dynamic_samples, strategy=strategy, padding_ratio=padding)
            
            min_shape = tuple(fixed_dims + [min_d])
            opt_shape = tuple(fixed_dims + [opt_d])
            max_shape = tuple(fixed_dims + [max_d])
            
            prof_dict[tensor_name] = {
                "min": min_shape,
                "opt": opt_shape,
                "max": max_shape
            }
        profiles.append(prof_dict)
        
    return profiles
