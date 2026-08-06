import unittest
from minja_tools.decision import ToolDecisionEngine


class TestToolDecision(unittest.TestCase):
    def test_decision_logic(self):
        engine = ToolDecisionEngine(["abs", "length"])
        self.assertTrue(engine.needs_jinja("{{ x | custom_filter }}", True))
        self.assertFalse(engine.needs_jinja("{{ x | abs }}", True))


def test_regression():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestToolDecision)
    result = unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():
        raise AssertionError("Regression tests failed")
