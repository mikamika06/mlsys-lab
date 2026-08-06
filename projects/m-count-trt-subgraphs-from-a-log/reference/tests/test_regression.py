import unittest
from trtlog.parser import parse_log
from trtlog.bench import compare_eps, simulate_cache_sessions


class TestTRTLog(unittest.TestCase):
    def test_parser(self):
        log = "I [TensorRTExecutionProvider] Subgraph 0: 10 nodes partitioned.\n"
        res = parse_log(log)
        self.assertEqual(res["count"], 1)
        self.assertEqual(res["subgraphs"][0]["nodes"], 10)

    def test_bench(self):
        c = [10.0, 12.0]
        t = [5.0, 6.0]
        res = compare_eps(c, t)
        self.assertAlmostEqual(res["latency_ratio"], 2.0)

    def test_cache(self):
        logs = ["I timing cache loaded"]
        res = simulate_cache_sessions(logs)
        self.assertTrue(res[0])


if __name__ == "__main__":
    unittest.main()
