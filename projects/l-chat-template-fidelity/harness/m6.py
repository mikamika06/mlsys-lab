import importlib

import ref

BROKEN = [
    "{{ if .Messages }}no end",
    "{{ end }}",
    "{{ range .Messages }}{{ end }}{{ end }}",
    "{{ nosuchfunction }}",
    "{{ .Messages",
]


def _strip_render(real):
    def render(src, data, funcs=None):
        return real(src, data, funcs).strip()
    return render


def check(workdir):
    from gotmpl import TemplateError, render

    out = {"rejects_malformed": 0.0, "suite_passes": 0.0, "suite_catches": 0.0}

    rejected = 0
    for src in BROKEN:
        try:
            render(src, ref.inputs()["01_single_user"], ref.funcs())
        except TemplateError:
            rejected += 1
        except Exception:  # noqa: BLE001
            pass
    out["rejects_malformed"] = rejected / len(BROKEN)

    mod = importlib.import_module("tests.test_templates")
    if not hasattr(mod, "run_suite"):
        return out
    passed, failed = mod.run_suite(render, ref.funcs(), ref.TEMPLATES)
    if failed == 0 and passed >= 300:
        out["suite_passes"] = 1.0
    # A suite that reports success no matter what is not a suite. Hand it a
    # renderer that trims the output and it has to notice.
    p2, f2 = mod.run_suite(_strip_render(render), ref.funcs(), ref.TEMPLATES)
    if f2 > 0:
        out["suite_catches"] = 1.0
    return out
