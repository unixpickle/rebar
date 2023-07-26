from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def control_variate_losses(
    u1: torch.Tensor,
    u2: torch.Tensor,
    pre_logits: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    lam: torch.Tensor,
) -> torch.Tensor:
    """
    :param u1: a uniform tensor of shape [N x n_latent, n_vocab]
    :param u2: a uniform tensor of shape [N x n_latent, n_vocab]
    :param pre_logits: a tensor of shape [N x n_latent x n_vocab]
    :param decoder: the decoder module
    :param loss_fn: function mapping [N x n_latent x n_vocab] to [N] loss
    :param lam: the temperature control for Gumbel relaxation.
    :return: a loss tensor of shape [N]
    """
    z = gumbel(u1, pre_logits)
    maxes = hard_threshold(z)
    log_probs = (
        F.log_softmax(pre_logits, dim=-1)
        .gather(-1, maxes.argmax(-1, keepdim=True))
        .flatten(1)
        .sum(1)
    )
    z_tilde = conditional_gumbel_logits(u2, maxes, pre_logits)
    soft_out = loss_fn(soft_threshold(z, lam))
    soft_out_tilde = loss_fn(soft_threshold(z_tilde, lam))
    return -soft_out_tilde.detach() * log_probs + soft_out - soft_out_tilde


def reinforce_losses(
    u1: torch.Tensor,
    pre_logits: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    sample_baseline: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = gumbel(u1, pre_logits)
    maxes = hard_threshold(z)
    log_probs = (
        F.log_softmax(pre_logits, dim=-1)
        .gather(-1, maxes.argmax(-1, keepdim=True))
        .flatten(1)
        .sum(1)
    )
    with torch.no_grad():
        loss = loss_fn(maxes)
        if sample_baseline:
            loss -= loss_fn(hard_threshold(gumbel(torch.rand_like(u1), pre_logits)))
    return loss * log_probs, loss


def gumbel(u1: torch.Tensor, pre_logits: torch.Tensor):
    log_probs = F.log_softmax(pre_logits, dim=-1)
    return log_probs - safe_log(-safe_log(u1))


def hard_threshold(gumbel_out: torch.Tensor) -> torch.Tensor:
    results = torch.zeros_like(gumbel_out)
    results.scatter_(
        -1, gumbel_out.argmax(-1, keepdim=True), torch.ones_like(results[..., :1])
    )
    return results


def soft_threshold(gumbel_out: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return F.softmax(gumbel_out / lam, dim=-1)


def conditional_gumbel_logits(
    u2: torch.Tensor, maxes: torch.Tensor, pre_logits: torch.Tensor
) -> torch.Tensor:
    """
    :param u2: a U[0, 1] sample batch.
    :param maxes: a batch of one-hots from hard_threshold.
    :param pre_logits: the same argument passed to gumbel().
    """
    mask = maxes.bool()
    log_probs = F.log_softmax(pre_logits, dim=-1)
    logs = safe_log(u2)
    log_vk = logs.gather(-1, maxes.argmax(-1, keepdim=True))

    term1 = safe_log(-logs)

    # -(-logs / probs - log_vk).log()
    left = term1 - log_probs
    right = safe_log(-log_vk)
    term2 = torch.logsumexp(torch.stack([left, right.expand_as(left)]), dim=0)

    return torch.where(mask, -term1, -term2)


def safe_log(xs: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.where(xs < 1e-8, 1, xs))
