from profdebug.compat import check_compat

def test_compat_cases():
    assert check_compat("535.54.03", "2023.2") is True
    assert check_compat("535.54.03", "2020.1") is False
