import numpy as np
from eagle.head import DraftHead
from eagle.integration import EagleEngine


def get_reference_engine():
    head = DraftHead(64, 200, seed=123)
    engine = EagleEngine(64, 200, head)
    return engine
