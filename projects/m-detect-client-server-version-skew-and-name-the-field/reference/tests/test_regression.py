from runnerdiag.skew import check_version_skew


def test_version_skew_detection():
    res_match = check_version_skew("0.1.30", {"version": "0.1.30"})
    assert res_match["has_skew"] is False
    assert res_match["proving_field"] == "version"

    res_skew = check_version_skew("0.1.30", {"version": "0.1.32"})
    assert res_skew["has_skew"] is True
    assert res_skew["proving_field"] == "version"

    res_missing = check_version_skew("0.1.30", {})
    assert res_missing["has_skew"] is True
    assert res_missing["proving_field"] == "version"
