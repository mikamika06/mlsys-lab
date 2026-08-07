import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from vllm_metrics.parser import parse_scrape
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"Failed to import parse_scrape: {e}"}

    max_err = 0.0
    for i, text in enumerate(ref.TEST_SCRAPES):
        want = ref.parse_scrape(text)
        try:
            got = parse_scrape(text)
        except Exception as e:
            return {
                "rel_err": 1.0,
                "_note": f"parse_scrape raised on scrape {i}: {e}",
            }

        if set(got.keys()) != set(want.keys()):
            return {
                "rel_err": 1.0,
                "_note": f"Scrape {i}: key mismatch got {list(got.keys())} want {list(want.keys())}",
            }

        for fam_key, want_fam in want.items():
            got_fam = got[fam_key]
            if (
                got_fam.get("name") != want_fam.get("name")
                or got_fam.get("type") != want_fam.get("type")
            ):
                return {
                    "rel_err": 1.0,
                    "_note": f"Scrape {i} family {fam_key} header mismatch",
                }
            got_samples = got_fam.get("samples", [])
            want_samples = want_fam.get("samples", [])
            if len(got_samples) != len(want_samples):
                return {
                    "rel_err": 1.0,
                    "_note": f"Scrape {i} family {fam_key} sample length mismatch",
                }
            for s_got, s_want in zip(got_samples, want_samples):
                if (
                    s_got.get("name") != s_want.get("name")
                    or s_got.get("labels") != s_want.get("labels")
                ):
                    return {
                        "rel_err": 1.0,
                        "_note": f"Scrape {i} sample metadata mismatch",
                    }
                v_got = float(s_got.get("value", 0.0))
                v_want = float(s_want.get("value", 0.0))
                diff = abs(v_got - v_want) / (1.0 + abs(v_want))
                if diff > max_err:
                    max_err = diff

    return {"rel_err": max_err}
