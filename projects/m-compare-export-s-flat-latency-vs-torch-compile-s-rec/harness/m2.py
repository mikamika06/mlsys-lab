import harness.ref as ref


def check(workdir):
    import exportbench.artifact as artifact

    out = {"deserialization_handled": 0.0, "valid_artifact_loaded": 0.0}

    graph_data = ref.sample_graph_data()
    weights = ref.sample_weights()

    try:
        valid_bytes = artifact.serialize_export_artifact(graph_data, weights)
        parsed_graph, parsed_weights = artifact.deserialize_export_artifact(valid_bytes)
        if parsed_graph == graph_data and len(parsed_weights) == weights.tobytes().__len__():
            out["valid_artifact_loaded"] = 1.0
        else:
            out["_note"] = "Valid artifact roundtrip mismatch"
            return out
    except Exception as e:
        out["_note"] = f"Valid artifact failed to parse: {type(e).__name__}: {str(e)[:120]}"
        return out

    corrupted_cases = [
        valid_bytes[:8],
        b"BADH" + valid_bytes[4:],
        valid_bytes[: len(valid_bytes) - 10],
    ]

    handled_count = 0
    for payload in corrupted_cases:
        try:
            artifact.deserialize_export_artifact(payload)
        except ValueError:
            handled_count += 1
        except Exception as e:
            out["_note"] = f"Corrupted payload raised wrong exception: {type(e).__name__}"
            return out

    if handled_count == len(corrupted_cases):
        out["deserialization_handled"] = 1.0
    else:
        out["_note"] = f"Handled {handled_count}/{len(corrupted_cases)} corrupted payload cases"

    return out
