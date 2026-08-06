class GKDConfig:
    def __init__(self, beta=0.0, lmbda=1.0, temperature=1.0):
        self.beta = beta
        self.lmbda = lmbda
        self.temperature = temperature


def compute_gkd_step(student_logits, teacher_logits, config):
    raise NotImplementedError
