import unittest
from backperf.cmake import enumerate_flags
from backperf.throughput import compare_throughput
from backperf.repack import analyze_size_invariance

class TestBuildBackendRegression(unittest.TestCase):
    def test_cpu_flags(self):
        flags = enumerate_flags("cpu")
        self.assertIn("-DGGML_STATIC=ON", flags)

    def test_gpu_flags(self):
        flags = enumerate_flags("gpu")
        self.assertIn("-DGGML_CUDA=ON", flags)

    def test_size_invariance(self):
        fix = {"size_before_bytes": 1000, "size_after_bytes": 1000, "reason": "test"}
        res = analyze_size_invariance(fix)
        self.assertTrue(res["unchanged"])

def test_run_all():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBuildBackendRegression)
    result = unittest.TextTestRunner().run(suite)
    if not result.wasSuccessful():
        raise AssertionError("tests failed")
