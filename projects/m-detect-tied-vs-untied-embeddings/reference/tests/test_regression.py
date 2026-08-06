import unittest
from ggufmap.moe import build_moe_inventory


class TestMoEInventory(unittest.TestCase):

    def test_inventory_structure(self):
        names = [
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
        ]
        inv = build_moe_inventory(names)
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0]["layer"], 0)
        self.assertEqual(inv[0]["experts"], [0, 1])


if __name__ == "__main__":
    unittest.main()
