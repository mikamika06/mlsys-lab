import ref


def check(workdir):
    from mlx_vlm_edge.error_reproducer import reproduce_error, explain_error

    out = {"error_reproduced": 0.0, "explanation_valid": 0.0}
    text_config = {"model_type": "llama", "hidden_size": 4096}

    try:
        reproduce_error(text_config, "Hello world")
        out["_note"] = "reproduce_error did not raise expected error on text-only config"
    except (KeyError, ValueError, TypeError):
        out["error_reproduced"] = 1.0
    except Exception as e:
        out["_note"] = f"reproduce_error raised unexpected exception type: {type(e).__name__}"

    try:
        expl = explain_error()
        if isinstance(expl, str) and len(expl) > 20:
            out["explanation_valid"] = 1.0
        else:
            out["_note"] = "explanation_valid returned non-string or too short description"
    except Exception as e:
        out["_note"] = f"explain_error failed: {str(e)}"

    return out
