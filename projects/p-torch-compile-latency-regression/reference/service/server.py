import torch

from .model import Classifier
from .preprocess import normalise

BATCH_SCHEDULE = [1, 2, 3, 4, 5, 6, 7, 8]


class Service:
    def __init__(self, compiled: bool = False):
        self.model = Classifier().eval()
        self.fn = torch.compile(self.model, dynamic=True) if compiled else self.model

    @torch.no_grad()
    def handle(self, batch):
        return self.fn(normalise(batch))
