import unittest
from profiler_bench.overhead import measure_overhead_ratio
from profiler_bench.ranking import rank_profiler_options

class TestProfilerBench(unittest.TestCase):
    def test_overhead_positive(self):
        ratio = measure_overhead_ratio()
        self.assertGreater(ratio, 0.0)

    def test_ranking_order(self):
        ranking = rank_profiler_options()
        self.assertEqual(ranking[0], "with_stack")
        self.assertEqual(ranking[-1], "with_flops")

if __name__ == "__main__":
    unittest.main()
