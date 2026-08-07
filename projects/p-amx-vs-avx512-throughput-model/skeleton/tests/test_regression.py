def test_selection():
    from amx_model.model import select_best_isa
    res = select_best_isa(256, 256, 256, "int8")
    assert res == "amx"
