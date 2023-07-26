import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self, *, n_vocab: int, n_latent: int, d_emb: int, device: torch.device
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_latent = n_latent
        self.d_emb = d_emb
        self.device = device
        self.unemb = nn.Linear(n_vocab * n_latent, d_emb, device=device)
        nn.init.normal_(self.unemb.weight, std=1 / math.sqrt(n_latent))

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        res = self.unemb(x.flatten(1))
        return F.log_softmax(res, dim=-1)


class MLPDecoder(Decoder):
    def __init__(self, *, channels: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        layers = []
        n_in = self.d_emb
        for n_out in channels:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(n_in, n_out, device=self.device))
            n_in = n_out
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.unembed(x))
