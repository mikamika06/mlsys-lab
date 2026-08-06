"""Loss convergence and parity verification."""

def verify_loss_convergence(pt_losses, mlx_losses, max_relative_diff=0.1):
    if len(pt_losses) != len(mlx_losses) or not pt_losses:
        return {"comparable": False, "max_diff": float("inf"), "mean_diff": float("inf")}

    diffs = [abs(p - m) for p, m in zip(pt_losses, mlx_losses)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    pt_final = pt_losses[-1]
    mlx_final = mlx_losses[-1]
    final_diff = abs(pt_final - mlx_final) / max(abs(pt_final), 1e-8)

    comparable = final_diff <= max_relative_diff
    return {
        "comparable": comparable,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "pt_final_loss": pt_final,
        "mlx_final_loss": mlx_final
    }
