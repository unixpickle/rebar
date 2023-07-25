from typing import Callable

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .rebar import control_variate_losses, gumbel, hard_threshold, reinforce_losses


class RebarModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, init_lam: float = 0.1):
        self.encoder = encoder
        self.decoder = decoder
        self.device = encoder.device
        self.eta_arg = nn.Parameter(
            torch.zeros(len(list(encoder.parameters())), device=self.device)
        )
        self.lam_arg = nn.Parameter(
            torch.log(torch.zeros(device=self.device) + init_lam)
        )

    def backward(
        self,
        inputs: torch.Tensor,
        output_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        def loss_fn(x: torch.Tensor):
            return output_loss_fn(self.decoder(x))

        for param in self.decoder.parameters():
            param.requires_grad_(False)

        enc_out = self.encoder(inputs)
        u1 = torch.rand_like(enc_out)
        u2 = torch.rand_like(enc_out)

        # Get gradients for both the encoder and the decoder.
        reinforce = reinforce_losses(u1=u1, pre_logits=enc_out, loss_fn=loss_fn)
        reinforce.mean().backward(retain_graph=True)

        control = control_variate_losses(
            u1=u1,
            u2=u2,
            pre_logits=enc_out,
            loss_fn=loss_fn,
            lam=self.lam_arg.exp().detach(),
        )
        grads = torch.autograd.grad(control.mean(), list(self.encoder.parameters()))
        for param, grad, eta in zip(
            self.encoder.parameters(), grads, self.eta_arg.detach().sigmoid() * 2
        ):
            param.grad.add_(grad * eta)

        for param in self.decoder.parameters():
            param.requires_grad_(True)

        hard_out = hard_threshold(gumbel(u1=u1, pre_logits=enc_out))
        dec_loss = loss_fn(hard_out)
        dec_loss.backward()
