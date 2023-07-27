from contextlib import contextmanager
from random import sample
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .rebar import control_variate_losses, gumbel, hard_threshold, reinforce_losses


class RebarModel(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        init_eta: float = 1.0,
        init_lam: float = 0.1,
        sample_baseline: bool = False,
    ):
        super().__init__()
        assert init_eta > 0 and init_eta < 2
        assert init_lam > 0
        self.encoder = encoder
        self.decoder = decoder
        self.device = encoder.device
        self.eta_arg = nn.Parameter(
            torch.logit(
                torch.zeros(len(list(encoder.parameters())), device=self.device)
                + init_eta / 2,
            )
        )
        self.lam_arg = nn.Parameter(
            torch.log(torch.zeros((), device=self.device) + init_lam)
        )
        self.sample_baseline = sample_baseline

    def eta(self) -> torch.Tensor:
        return self.eta_arg.sigmoid() * 2

    def lam(self) -> torch.Tensor:
        return self.lam_arg.exp()

    def backward(
        self,
        inputs: torch.Tensor,
        output_loss_fn: Callable[[torch.Tensor], torch.Tensor],
        entropy_coeff: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        def loss_fn(x: torch.Tensor):
            return output_loss_fn(self.decoder(x))

        with toggle_grads(self.decoder.parameters(), False):
            dist = self.encoder(inputs)
            entropies = dist.entropy().flatten(1).sum(1)
            if entropy_coeff:
                (entropies * entropy_coeff).mean().backward(retain_graph=True)
            enc_out = dist.logits
            u1 = torch.rand_like(enc_out)
            u2 = torch.rand_like(enc_out)

            # Get gradients for both the encoder and the decoder.
            reinforce, losses = reinforce_losses(
                u1=u1,
                pre_logits=enc_out,
                loss_fn=loss_fn,
                sample_baseline=self.sample_baseline,
            )
            reinforce.mean().backward(retain_graph=True)

            control = control_variate_losses(
                u1=u1,
                u2=u2,
                pre_logits=enc_out,
                loss_fn=loss_fn,
                lam=self.lam().detach(),
            )
            grads = torch.autograd.grad(control.mean(), list(self.encoder.parameters()))
            for param, grad, eta in zip(
                self.encoder.parameters(), grads, self.eta().detach()
            ):
                param.grad.add_(grad * eta)

        # This generates gradients for the decoder only
        hard_out = hard_threshold(gumbel(u1=u1, pre_logits=enc_out.detach()))
        dec_loss = loss_fn(hard_out)
        dec_loss.mean().backward()

        return dict(loss=losses.detach(), entropy=entropies.detach())

    def variance_backward(
        self,
        inputs: torch.Tensor,
        output_loss_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the gradients of the variance w.r.t. the etas and lambda.
        If self.lam_arg does not require grad, then second derivatives will not
        be required for the computation.
        """
        params = list(self.encoder.parameters())
        all_grads = [[] for _ in range(len(params))]
        with toggle_grads(list(self.decoder.parameters()), False):
            for i, single_input in enumerate(inputs):
                grads = self._encoder_grads(
                    single_input[None],
                    self._unbatched_loss_function(output_loss_fn, len(inputs), i),
                    create_graph=self.lam_arg.requires_grad,
                )
                for j, g in enumerate(grads):
                    all_grads[j].append(g)
        result = 0.0
        for gs in all_grads:
            result = result + torch.stack(gs).flatten(1).var(0).sum()

        # Block gradients to main parameters.
        if self.lam_arg.requires_grad:
            eta_grad, lam_grad = torch.autograd.grad(
                result, [self.eta_arg, self.lam_arg]
            )
            self.eta_arg.backward(gradient=eta_grad)
            self.lam_arg.backward(gradient=lam_grad)
        else:
            (eta_grad,) = torch.autograd.grad(result, [self.eta_arg])
            self.eta_arg.backward(eta_grad)
        return result.detach()

    def _encoder_grads(
        self,
        inputs: torch.Tensor,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        create_graph: bool = False,
    ) -> List[torch.Tensor]:
        enc_out = self.encoder(inputs).logits
        u1 = torch.rand_like(enc_out)
        u2 = torch.rand_like(enc_out)

        # Get gradients for both the encoder and the decoder.
        reinforce, _losses = reinforce_losses(
            u1=u1,
            pre_logits=enc_out,
            loss_fn=loss_fn,
            sample_baseline=self.sample_baseline,
        )
        reinforce_grads = torch.autograd.grad(
            reinforce.mean(),
            list(self.encoder.parameters()),
            create_graph=create_graph,
            retain_graph=True,
        )
        control = control_variate_losses(
            u1=u1,
            u2=u2,
            pre_logits=enc_out,
            loss_fn=loss_fn,
            lam=self.lam(),
        )
        control_grads = torch.autograd.grad(
            control.mean(), list(self.encoder.parameters()), create_graph=create_graph
        )
        return [
            reinforce_grad + control_grad * eta
            for reinforce_grad, control_grad, eta in zip(
                reinforce_grads, control_grads, self.eta()
            )
        ]

    def _unbatched_loss_function(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        orig_count: int,
        index: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def new_loss_fn(x: torch.Tensor) -> torch.Tensor:
            outputs = self.decoder(x)
            rep_outputs = outputs.repeat(orig_count, *([1] * (len(outputs.shape) - 1)))
            return loss_fn(rep_outputs)[index : index + 1]

        return new_loss_fn


@contextmanager
def toggle_grads(params: Iterable[nn.Parameter], enabled: bool):
    params = list(params)
    backups = []
    for p in params:
        backups.append(p.requires_grad)
        p.requires_grad_(enabled)
    try:
        yield
    finally:
        for p, backup in zip(params, backups):
            p.requires_grad_(backup)
