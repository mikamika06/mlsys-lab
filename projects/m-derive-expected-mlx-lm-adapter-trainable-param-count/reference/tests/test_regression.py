from mlx_lora_util.derive import expected_adapter_parameters
from mlx_lora_util.verify import verify_safetensors_shapes
from mlx_lora_util.train import simulate_training_loss


def test_derivation_correctness():
    cfg = {"hidden_size": 512, "num_hidden_layers": 2, "num_attention_heads": 8, "num_key_value_heads": 8, "intermediate_size": 1024}
    val = expected_adapter_parameters(cfg, 4, ["q_proj", "v_proj"])
    assert val > 0


def test_training_loss_decreases():
    losses = simulate_training_loss(10, 2.5)
    assert len(losses) == 10
    assert losses[-1] < losses[0]
