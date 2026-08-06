import torch
import torch.library


def raw_smooth_relumix(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Raw Python implementation of smooth relumix."""
    return torch.where(x > 0, x * alpha, torch.exp(x) - 1.0)


@torch.library.custom_op("customop::smooth_relumix", mutates_args=())
def smooth_relumix(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Smooth relumix custom operation."""
    return raw_smooth_relumix(x, alpha)


@smooth_relumix.register_fake
def _(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.empty_like(x)


def register_custom_op():
    """Ensure custom op registration is present."""
    return smooth_relumix


def validate_op_schema(x: torch.Tensor, alpha: float) -> bool:
    """Run opcheck on the registered custom op."""
    try:
        torch.library.opcheck(smooth_relumix, (x, alpha))
        return True
    except Exception:
        return False
