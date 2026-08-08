def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import quant
    import ref
    model, x = ref.get_fixture()
    sens_ref = ref.measure_sensitivity(model, x, [32, 8, 4, 2])
    sens_stu = quant.measure_sensitivity(model, x, [32, 8, 4, 2])
    match = 1.0 if ref.dict_close(sens_ref, sens_stu) else 0.0
    return {"mse_match": match}
