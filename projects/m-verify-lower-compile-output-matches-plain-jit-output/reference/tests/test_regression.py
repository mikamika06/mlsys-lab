import numpy as np
import pytest
from jaxinspect.verify import verify_compile_vs_jit
from jaxinspect.ir import analyze_stablehlo_ir


def test_verify_compile_vs_jit_max_error():
    aot = [{"data": np.array([1.0, 2.0])}, {"data": np.array([3.0, 4.0])}]
    jit = [{"data": np.array([1.0, 2.0001])}, {"data": np.array([3.0, 4.0])}]
    res = verify_compile_vs_jit(aot, jit)
    assert res["max_abs_err"] > 0.0
    assert not res["is_close"]


def test_analyze_stablehlo_ir_counts():
    ir = """
    module {
      func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
        %0 = stablehlo.abs %arg0 : tensor<2x2xf32>
        %1 = stablehlo.add %0, %0 : tensor<2x2xf32>
        %2 = stablehlo.add %1, %0 : tensor<2x2xf32>
        return %2 : tensor<2x2xf32>
      }
    }
    """
    res = analyze_stablehlo_ir(ir)
    assert res["op_counts"]["stablehlo.add"] == 2
    assert res["op_counts"]["stablehlo.abs"] == 1
    assert res["unique_ops"] == ["stablehlo.abs", "stablehlo.add"]
