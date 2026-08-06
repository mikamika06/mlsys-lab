import ref


def roundtrip_export_load(model_spec):
    return ref.verify_roundtrip(model_spec)


def profile_artifact_sizes(model_spec):
    return ref.compute_artifact_sizes(model_spec)
