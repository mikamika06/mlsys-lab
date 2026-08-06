import mxfp4.analysis
import mxfp4.moe


def test_requantization_logic():
    cfg = {"source_format": "mxfp4", "target_format": "mxfp4", "block_size": 32, "target_block_size": 32}
    assert mxfp4.analysis.is_requantization_lossy(cfg) is True


def test_moe_share_logic():
    spec = {
        "layers": [
            {
                "type": "moe",
                "num_experts": 2,
                "expert_params": 64,
                "router_params": 32,
                "block_size": 32
            }
        ]
    }
    share = mxfp4.moe.compute_mxfp4_share(spec)
    assert 0.0 <= share <= 1.0
