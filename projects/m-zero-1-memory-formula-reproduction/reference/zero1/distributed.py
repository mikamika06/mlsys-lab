import torch


class ToyZeRO1Optimizer:

    def __init__(self, params, lr=1e-3, world_size=1, rank=0):
        self.params = list(params)
        self.lr = lr
        self.world_size = world_size
        self.rank = rank

        self.total_numel = sum(p.numel() for p in self.params)
        self.shard_size = (self.total_numel + world_size - 1) // world_size

        start = rank * self.shard_size
        end = min(start + self.shard_size, self.total_numel)
        self.local_numel = max(0, end - start)

        flat = torch.cat([p.detach().flatten() for p in self.params])
        if start < self.total_numel:
            self.fp32_shard = flat[start:end].to(torch.float32).clone()
        else:
            self.fp32_shard = torch.empty(0, dtype=torch.float32)

        self.exp_avg = torch.zeros_like(self.fp32_shard)
        self.exp_avg_sq = torch.zeros_like(self.fp32_shard)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_num = 0

    def step(self, grads):
        self.step_num += 1
        flat_grad = torch.cat([g.flatten() for g in grads])

        start = self.rank * self.shard_size
        end = min(start + self.shard_size, self.total_numel)

        if start < self.total_numel:
            grad_shard = flat_grad[start:end].to(torch.float32)

            self.exp_avg.mul_(self.beta1).add_(grad_shard, alpha=1.0 - self.beta1)
            self.exp_avg_sq.mul_(self.beta2).addcmul_(
                grad_shard, grad_shard, value=1.0 - self.beta2
            )

            bias_corr1 = 1.0 - self.beta1**self.step_num
            bias_corr2 = 1.0 - self.beta2**self.step_num

            step_size = self.lr / bias_corr1
            denom = (self.exp_avg_sq.sqrt() / (bias_corr2**0.5)).add_(self.eps)

            self.fp32_shard.addcdiv_(self.exp_avg, denom, value=-step_size)
            updated_shard = self.fp32_shard.to(self.params[0].dtype)
        else:
            updated_shard = torch.empty(0, dtype=self.params[0].dtype)

        padded_shard = torch.zeros(self.shard_size, dtype=self.params[0].dtype)
        if len(updated_shard) > 0:
            padded_shard[: len(updated_shard)] = updated_shard

        gathered = [
            torch.zeros(self.shard_size, dtype=self.params[0].dtype)
            for _ in range(self.world_size)
        ]
        if torch.distributed.is_initialized():
            torch.distributed.all_gather(gathered, padded_shard)
        else:
            gathered[self.rank] = padded_shard

        full_flat = torch.cat(gathered)[: self.total_numel]

        offset = 0
        for p in self.params:
            numel = p.numel()
            p.data.copy_(full_flat[offset : offset + numel].view_as(p))
            offset += numel

    def get_rank_state_bytes(self):
        bytes_per_elem = 4
        return (
            self.fp32_shard.numel() + self.exp_avg.numel() + self.exp_avg_sq.numel()
        ) * bytes_per_elem
