import time
import torch

def measure_checkpoint_overhead(model, inputs):
    model.eval()

    def run_pass(m, x):
        optimizer = torch.optim.SGD(m.parameters(), lr=0.01)
        optimizer.zero_grad()
        start = time.perf_counter()
        out = m(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()
        end = time.perf_counter()
        return end - start

    t_baseline = run_pass(model, inputs)
    return float(t_baseline)
