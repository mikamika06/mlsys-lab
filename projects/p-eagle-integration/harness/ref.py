import numpy as np
from eagle.head import DraftHead
from eagle.sampler import DraftSampler
from eagle.integration import EagleIntegration


def get_reference_objects():
    head = DraftHead(4, 10)
    sampler = DraftSampler(1.0)
    integration = EagleIntegration(None, head)
    return head, sampler, integration
