from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class Encoder(nn.Module):
    def __init__(
        self, *, n_vocab: int, n_latent: int, d_emb: int, device: torch.device
    ):
        self.n_vocab = n_vocab
        self.n_latent = n_latent
        self.d_emb = d_emb
        self.device = device
        self.emb = nn.Linear(d_emb, n_vocab * n_latent, device=device)

    def embed(self, x: torch.Tensor) -> Categorical:
        res = self.emb(x).view(-1, self.n_latent, self.n_vocab)
        return Categorical(logits=res)


class MLPEncoder(nn.Module):
    def __init__(self, *, channels: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        layers = []
        n_in = channels[0]
        for n_out in list(channels[1:]) + [self.d_emb]:
            layers.append(nn.Linear(n_in, n_out, device=self.device))
            layers.append(nn.ReLU())
            n_in = n_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Categorical:
        return self.embed(self.mlp(x))
