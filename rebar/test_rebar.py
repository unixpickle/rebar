import torch
import torch.nn.functional as F

from rebar.rebar import conditional_gumbel_logits, gumbel, hard_threshold


def test_gumbel_distribution():
    torch.manual_seed(0)
    n = 1000000
    logits = torch.tensor([[1.0, 2.0, -0.2]]).repeat(n, 1)
    samples = hard_threshold(gumbel(torch.rand_like(logits), logits))
    actual = samples.mean(0)
    expected = F.softmax(logits, dim=-1)[0]
    assert (actual - expected).abs().max().item() < 1e-2, f"{actual=} {expected=}"


def test_conditional_gumbel():
    torch.manual_seed(0)
    n = 5000000
    logits = torch.tensor([[1.0, 2.0, -0.2]]).repeat(n, 1)
    u1 = torch.rand_like(logits)
    u2 = torch.rand_like(logits)
    z = gumbel(u1, logits)
    b = hard_threshold(z)
    z_tilde = conditional_gumbel_logits(u2, b, logits)

    hist_args = dict(bins=20, min=-7, max=7)
    for value in range(3):
        mask = b.argmax(-1) == value
        zs = z[mask]
        zs_tilde = z_tilde[mask]
        for i in range(3):
            hist = torch.histc(zs[:, i], **hist_args).float() / n
            hist_tilde = torch.histc(zs_tilde[:, i], **hist_args).float() / n
            assert (
                hist - hist_tilde
            ).abs().max().item() < 1e-3, f"{hist.tolist()=} {hist_tilde.tolist()=}"
