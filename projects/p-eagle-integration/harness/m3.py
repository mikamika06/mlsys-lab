import ref


def check(workdir):
    from eagle.integration import EagleIntegration
    from eagle.head import DraftHead
    m = {"acceptance_measured": 0.0}
    try:
        head = DraftHead(4, 10)
        integ = EagleIntegration(None, head)
        acc = integ.measure_acceptance(["test"])
        if isinstance(acc, (int, float)) and acc > 0:
            m["acceptance_measured"] = 1.0
    except Exception:
        pass
    return m
