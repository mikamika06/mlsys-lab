import tempfile
from typing import Any, Dict, Tuple
import torch
import torch.export


def verify_graph_signature(
    mod: torch.nn.Module,
    ep: torch.export.ExportedProgram,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any] | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    if kwargs is None:
        kwargs = {}

    sig = ep.graph_signature
    mod_params = dict(mod.named_parameters(remove_duplicate=False))
    mod_buffers = dict(mod.named_buffers(remove_duplicate=False))

    ep_param_names = set(sig.inputs_to_parameters.values())
    ep_buffer_names = set(sig.inputs_to_buffers.values())

    params_ok = set(mod_params.keys()) == ep_param_names
    buffers_ok = set(mod_buffers.keys()) == ep_buffer_names

    param_shapes_ok = True
    for fqn in ep_param_names:
        if fqn in mod_params:
            spec_tensor = ep.state_dict.get(fqn)
            if spec_tensor is not None and spec_tensor.shape != mod_params[fqn].shape:
                param_shapes_ok = False
                break

    buffer_shapes_ok = True
    for fqn in ep_buffer_names:
        if fqn in mod_buffers:
            spec_tensor = ep.state_dict.get(fqn)
            if spec_tensor is not None and spec_tensor.shape != mod_buffers[fqn].shape:
                buffer_shapes_ok = False
                break

    user_inputs_count = len(sig.user_inputs)
    expected_inputs = len(args) + len(kwargs)
    inputs_ok = user_inputs_count == expected_inputs

    valid = params_ok and buffers_ok and param_shapes_ok and buffer_shapes_ok and inputs_ok

    details = {
        "params_ok": params_ok,
        "buffers_ok": buffers_ok,
        "param_shapes_ok": param_shapes_ok,
        "buffer_shapes_ok": buffer_shapes_ok,
        "inputs_ok": inputs_ok,
        "num_user_inputs": user_inputs_count,
        "expected_user_inputs": expected_inputs,
    }
    return valid, details


def verify_roundtrip_equivalence(
    ep: torch.export.ExportedProgram,
    sample_args: Tuple[Any, ...],
    sample_kwargs: Dict[str, Any] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    if sample_kwargs is None:
        sample_kwargs = {}

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        path = f.name
        torch.export.save(ep, path)
        reloaded_ep = torch.export.load(path)

    orig_out = ep.module()(*sample_args, **sample_kwargs)
    loaded_out = reloaded_ep.module()(*sample_args, **sample_kwargs)

    if isinstance(orig_out, torch.Tensor):
        if not isinstance(loaded_out, torch.Tensor):
            return False
        return torch.allclose(orig_out, loaded_out, atol=atol, rtol=rtol)

    if isinstance(orig_out, (tuple, list)):
        if not isinstance(loaded_out, (tuple, list)) or len(orig_out) != len(loaded_out):
            return False
        return all(
            torch.allclose(a, b, atol=atol, rtol=rtol)
            for a, b in zip(orig_out, loaded_out)
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
        )

    return True


def inspect_strict_export_behavior(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if kwargs is None:
        kwargs = {}

    strict_success = False
    strict_error = None
    try:
        ep_strict = torch.export.export(mod, args, kwargs, strict=True)
        strict_success = True
    except Exception as e:
        strict_error = str(e)

    nonstrict_success = False
    nonstrict_error = None
    try:
        ep_nonstrict = torch.export.export(mod, args, kwargs, strict=False)
        nonstrict_success = True
    except Exception as e:
        nonstrict_error = str(e)

    return {
        "strict_success": strict_success,
        "strict_error": strict_error,
        "nonstrict_success": nonstrict_success,
        "nonstrict_error": nonstrict_error,
        "strict_has_mutations": len(ep_strict.graph_signature.buffers_to_mutate) > 0 if strict_success else False,
        "nonstrict_has_mutations": len(ep_nonstrict.graph_signature.buffers_to_mutate) > 0 if nonstrict_success else False,
    }
