import unittest
from exporttools.opset import enumerate_opset
from exporttools.matrix import validate_target_matrix
from exporttools.repair import repair_image_input

class TestRegression(unittest.TestCase):
    def test_opset_basic(self):
        spec = {"opset_version": 3, "ops": ["add", "conv", "add"]}
        res = enumerate_opset(spec)
        self.assertEqual(res["opset_version"], 3)
        self.assertEqual(res["frequencies"]["add"], 2)

    def test_matrix_basic(self):
        res = validate_target_matrix("iOS15", ["gelu"])
        self.assertFalse(res["valid"])
        self.assertIn("gelu", res["unsupported"])

    def test_repair_basic(self):
        spec = {}
        fixed = repair_image_input(spec)
        self.assertTrue(fixed["is_image"])
        self.assertEqual(fixed["color_space"], "RGB")
