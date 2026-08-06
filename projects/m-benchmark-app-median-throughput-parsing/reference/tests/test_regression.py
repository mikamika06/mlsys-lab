from bapp.pc import parse_top_5

def test_parse_top_5_sort_order():
    log = """[ INFO ] layerName                                          execStatus    layerType       execType                 realTime (ms)  cpuTime (ms)
[ INFO ] -------------------------------------------------------------------------------------------------------------------------------------
[ INFO ] L1                                                 EXECUTED      Convolution     jit_avx512               1.000          1.000
[ INFO ] L2                                                 EXECUTED      Convolution     jit_avx512               2.000          2.000
[ INFO ] L3                                                 EXECUTED      Convolution     jit_avx512               3.000          3.000
[ INFO ] L4                                                 EXECUTED      Convolution     jit_avx512               4.000          4.000
[ INFO ] L5                                                 EXECUTED      Convolution     jit_avx512               5.000          5.000
[ INFO ] L6                                                 EXECUTED      Convolution     jit_avx512               6.000          6.000
[ INFO ] Total time: 21.000     milliseconds"""
    
    res = parse_top_5(log)
    assert len(res) == 5
    assert res[0][0] == "L6"
    assert res[0][1] == 6.0
    assert res[-1][0] == "L2"
