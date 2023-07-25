import torch
import torch.nn.functional as F


def gumbel(uniforms: torch.Tensor, pre_logits: torch.Tensor):
    log_probs = F.log_softmax(pre_logits, dim=-1)
    return log_probs - (-uniforms.log()).log()


def hard_threshold(gumbel_out: torch.Tensor) -> torch.Tensor:
    results = torch.zeros_like(gumbel_out)
    results.scatter_(
        -1, gumbel_out.argmax(-1, keepdim=True), torch.ones_like(results[..., :1])
    )
    return results


def soft_threshold(gumbel_out: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return F.softmax(gumbel_out / lam, dim=-1)


def conditional_gumbel_logits(
    other_uniforms: torch.Tensor, maxes: torch.Tensor, pre_logits: torch.Tensor
) -> torch.Tensor:
    """
    :param other_uniforms: a U[0, 1] sample batch.
    :param maxes: a batch of one-hots from hard_threshold.
    :param pre_logits: the same argument passed to gumbel().
    """
    mask = maxes.bool()
    probs = F.softmax(pre_logits, dim=-1)
    logs = other_uniforms.log()
    log_vk = logs.gather(-1, maxes.argmax(-1, keepdim=True))
    return torch.where(mask, -(-logs).log(), -(-logs / probs - log_vk).log())
