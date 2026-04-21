
import torch
from torch import nn
import math
import argparse
from adamw import AdamW
import time
import os

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1000, help="output dimension")
parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=-1.0, help="weight decay")
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--n_steps", type=int, default=20000, help="number of steps")
parser.add_argument("--dist", type=str, default="power", help="distribution of features")
parser.add_argument("--rho", type=float, default=0.0, help="feature correlation strength")
parser.add_argument("--block_size", type=int, default=10, help="block size for correlation")

args = parser.parse_args()

def probability(name, n):
    if name == "exponential":
        prob = torch.exp(-torch.arange(n) / 1000)
    elif name == "power":
        prob = torch.tensor([1.0 / i ** 1.2 for i in range(1, n+1)])
    elif name == "linear":
        prob = torch.tensor([(n-i) / n for i in range(n)])
    else:
        raise ValueError(f"Unknown distribution: {name}")
    return prob / prob.sum()

def generate_correlated_batch(batch_size, n, prob, rho, device, block_size=10):
    independent = torch.rand(batch_size, n, device=device)
    if rho == 0:
        u = (independent < prob).float()
        v = torch.rand(batch_size, n, device=device) * 2
        return u * v
    n_blocks = n // block_size
    block_signal = torch.rand(batch_size, n_blocks, device=device)
    block_signal = block_signal.repeat_interleave(block_size, dim=1)
    mixed = (1 - rho) * independent + rho * block_signal
    u = (mixed < prob).float()
    v = torch.rand(batch_size, n, device=device) * 2
    return u * v

m_ran = 2 ** torch.arange(3, 11)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prob = probability(args.dist, args.n).to(device)
criteria = nn.MSELoss()

class FeatureRecovery(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, m) / math.sqrt(m))
        self.b = nn.Parameter(torch.randn(n))
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x @ self.W @ self.W.T + self.b)

def get_lr(step, n_steps, warmup_ratio=0.05):
    warmup_steps = int(n_steps * warmup_ratio)
    step = step + 1
    min_lr = 0.05
    if step < warmup_steps:
        return step / warmup_steps
    return (1.0 - min_lr) * 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (n_steps - warmup_steps))) + min_lr

results = {}
Ws = {}
results["m_ran"] = m_ran
losses = torch.zeros(len(m_ran), args.n_steps)

for m_i, m in enumerate(m_ran):
    m = m.item()
    model = torch.compile(FeatureRecovery(args.n, m)).to(device)
    parameter_groups = [
        {"params": model.W, "weight_decay": args.weight_decay, "lr": args.lr * (8 / m) ** 0.25},
        {"params": model.b, "weight_decay": 0.0, "lr": 2.0 / m}
    ]
    optimizer = AdamW(parameter_groups)
    for pg in optimizer.param_groups:
        pg["init_lr"] = pg["lr"]

    t0 = time.perf_counter()
    for step in range(args.n_steps):
        optimizer.zero_grad(set_to_none=True)
        x = generate_correlated_batch(args.batch_size, args.n, prob, args.rho, device, args.block_size)
        for pg in optimizer.param_groups:
            pg["lr"] = pg["init_lr"] * get_lr(step, args.n_steps)
        y = model(x)
        loss = criteria(y, x) * 100
        losses[m_i, step] = loss.item() / 100
        loss.backward()
        optimizer.step()

    Ws[m_i] = model.W.detach().cpu()
    print(f"rho: {args.rho}, m: {m}, Loss: {losses[m_i,-1]:.2e}, Run time: {time.perf_counter()-t0:.2f}s")

results["losses"] = losses
results["W"] = Ws
results["rho"] = args.rho

output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)
torch.save(results, f"{output_dir}/corr_exp-rho{args.rho:.2f}_{args.dist}_{args.weight_decay:.2f}.pt")
print(f"Saved to {output_dir}/corr_exp-rho{args.rho:.2f}_{args.dist}_{args.weight_decay:.2f}.pt")
