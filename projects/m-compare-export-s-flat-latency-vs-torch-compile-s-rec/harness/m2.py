import ref


def check(workdir):
    from compilebench.artifact import (
        CorruptedArtifactError,
        InvalidMagicError,
        TruncatedArtifactError,
        deserialize_export_artifact,
        serialize_export_artifact,
    )

    out = {
        "deserializes_valid": 0.0,
        "catches_truncation": 0.0,
        "catches_corruption": 0.0,
        "catches_invalid_magic": 0.0,
    }

    try:
        spec = {"test": "data", "id": 123}
        art = serialize_export_artifact(spec)
        res = deserialize_export_artifact(art)
        if res == spec:
            out["deserializes_valid"] = 1.0
    except Exception as e:
        out["_note"] = f"Valid artifact failed: {e}"
        return out

    trunc_ok = True
    for tr in ref.TRUNCATED_ARTIFACTS:
        try:
            deserialize_export_artifact(tr)
            trunc_ok = False
            out["_note"] = "Failed to raise TruncatedArtifactError on truncated data"
            break
        except TruncatedArtifactError:
            pass
        except Exception as e:
            trunc_ok = False
            out["_note"] = f"Expected TruncatedArtifactError, got {type(e).__name__}"
            break
    if trunc_ok:
        out["catches_truncation"] = 1.0

    corr_ok = True
    for cr in ref.CORRUPTED_ARTIFACTS:
        try:
            deserialize_export_artifact(cr)
            corr_ok = False
            out["_note"] = "Failed to raise CorruptedArtifactError on corrupted payload/offset"
            break
        except CorruptedArtifactError:
            pass
        except Exception as e:
            corr_ok = False
            out["_note"] = f"Expected CorruptedArtifactError, got {type(e).__name__}"
            break
    if corr_ok:
        out["catches_corruption"] = 1.0

    magic_ok = True
    for mg in ref.INVALID_MAGIC_ARTIFACTS:
        try:
            deserialize_export_artifact(mg)
            magic_ok = False
            out["_note"] = "Failed to raise InvalidMagicError on bad magic header"
            break
        except InvalidMagicError:
            pass
        except Exception as e:
            magic_ok = False
            out["_note"] = f"Expected InvalidMagicError, got {type(e).__name__}"
            break
    if magic_ok:
        out["catches_invalid_magic"] = 1.0

    return out
