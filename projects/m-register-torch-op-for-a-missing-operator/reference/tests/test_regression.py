from milpass.audit import identify_pass
from milpass.fusion import fuse_conv_bn
import ref

def test_identify_pass_recognizes_fusion():
    g = ref.make_test_graph()
    fused, _ = fuse_conv_bn(g)
    p = identify_pass(g)
    assert p == "conv_bn_fusion"

def test_fusion_reduces_node_count():
    g = ref.make_test_graph()
    _, delta = fuse_conv_bn(g)
    assert delta < 0
