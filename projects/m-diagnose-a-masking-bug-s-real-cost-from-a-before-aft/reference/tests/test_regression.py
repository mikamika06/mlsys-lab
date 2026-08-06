import unittest
from ncu_diag.parser import parse_ncu_summary
from ncu_diag.analysis import compute_masking_cost


class TestNCUDiagnosisRegression(unittest.TestCase):
    def test_parser_basic(self):
        sample = "Metric Name,Metric Value\nsm__cycles_elapsed.avg,1,000\ninst_executed,500\n"
        res = parse_ncu_summary(sample)
        self.assertEqual(res["sm__cycles_elapsed.avg"], 1000.0)
        self.assertEqual(res["inst_executed"], 500.0)

    def test_analysis_basic(self):
        b = {"sm__cycles_elapsed.avg": 1000.0, "inst_executed": 500.0, "stall_mio_not_ready": 50.0}
        a = {"sm__cycles_elapsed.avg": 1200.0, "inst_executed": 600.0, "stall_mio_not_ready": 100.0}
        res = compute_masking_cost(b, a)
        self.assertAlmostEqual(res["cycle_diff"], 200.0)
        self.assertAlmostEqual(res["inst_diff"], 100.0)
        self.assertAlmostEqual(res["overhead_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
