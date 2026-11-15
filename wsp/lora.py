"""A trained reference for the conditional edits: LoRA on the MLP output matrices (the matrices the edits change).

The low-rank update is applied through a forward hook and is not merged into the weights. Training is plain
teacher forcing: the prompt without the instruction as input, the teacher's answer as target, loss on the answer only.
"""
import math
import time

import numpy as np
import torch


class LoRA:
    def __init__(self, lm, rank=4, seed=0):
        self.lm, self.rank = lm, rank
        for p in lm.model.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(seed)
        self.A = [torch.nn.Parameter((torch.randn(rank, lm.d_mlp, generator=g) / math.sqrt(lm.d_mlp)).to(lm.device)) for _ in range(lm.L)]
        self.B = [torch.nn.Parameter(torch.zeros(lm.d_model, rank, device=lm.device)) for _ in range(lm.L)]
        self.hooks = [m.register_forward_hook(lambda mod, inp, out, l=l: out + ((inp[0].to(self.A[l].dtype) @ self.A[l].T) @ self.B[l].T).to(out.dtype))
                      for l, m in enumerate(lm.mlp_out)]

    @property
    def n_parameters(self):
        return sum(a.numel() + b.numel() for a, b in zip(self.A, self.B))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def train(self, data, lr, epochs, batch=8, seed=0):
        """data: list of (prompt token ids, target token ids). Returns dict(steps, final_loss, seconds)."""
        lm = self.lm
        pad = next(iter(lm.end_ids))
        opt = torch.optim.AdamW(self.A + self.B, lr=lr, weight_decay=0.0)
        rng, t0, steps, loss = np.random.default_rng(seed), time.time(), 0, torch.tensor(float("nan"))
        for _ in range(epochs):
            perm = rng.permutation(len(data))
            for s in range(0, len(perm), batch):
                rows = [data[i] for i in perm[s:s + batch] if data[i][1]]
                if not rows:
                    continue
                T = max(len(p) + len(y) for p, y in rows)
                x = torch.full((len(rows), T), pad, device=lm.device)
                lab = torch.full((len(rows), T), -100, device=lm.device)
                for b, (p, y) in enumerate(rows):
                    x[b, :len(p) + len(y)] = torch.tensor(p + y, device=lm.device)
                    lab[b, len(p):len(p) + len(y)] = torch.tensor(y, device=lm.device)
                with torch.enable_grad():
                    logits = lm.head(lm.body(input_ids=x).last_hidden_state).float()
                    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                steps += 1
        return dict(steps=steps, final_loss=float(loss), seconds=round(time.time() - t0, 1))
