import pytest
import torch

from rebar.decoder import MLPDecoder
from rebar.encoder import MLPEncoder
from rebar.rebar_model import RebarModel


@pytest.mark.parametrize("dolambda", [False, True])
def test_variance(dolambda: bool):
    model = _testing_rebar_model()
    model.lam_arg.requires_grad_(dolambda)

    batch = torch.randn(100, 10)
    loss_fn = lambda x: (x - batch).pow(2).mean(1)
    torch.manual_seed(0)
    actual = model.variance_backward(batch, loss_fn).detach()

    torch.manual_seed(0)
    gradients = []
    for elem in batch:
        model.backward(elem[None], lambda x: (x - elem).pow(2).mean(1))
        gradients.append(
            torch.cat([x.grad.view(-1) for x in model.encoder.parameters()])
        )
        for x in model.encoder.parameters():
            x.grad = None
    expected = torch.stack(gradients).var(0).sum()

    assert actual.item() != 0

    assert (
        actual - expected
    ).abs().item() < 1e-3, f"{actual.item()=} {expected.item()=}"


def test_unbiased():
    torch.manual_seed(0)
    n_samples = 10000

    model = _testing_rebar_model()

    batch = torch.randn(2, 10).repeat(n_samples, 1)
    loss_fn = lambda x: (x - batch).pow(2).mean(1)
    model.backward(batch, loss_fn)
    with_rebar = torch.cat([x.grad.view(-1) for x in model.encoder.parameters()])
    for p in model.encoder.parameters():
        p.grad = None

    torch.manual_seed(0)
    with torch.no_grad():
        model.eta_arg.fill_(-100)
    model.backward(batch, loss_fn)
    without_rebar = torch.cat([x.grad.view(-1) for x in model.encoder.parameters()])

    noise = (with_rebar - without_rebar).pow(2).mean().item()
    signal = with_rebar.pow(2).mean().item()
    snr = signal / noise
    assert snr > 2, f"{signal=}:{noise=} ratio not good enough"

    cos_sim = torch.cosine_similarity(with_rebar, without_rebar, 0).item()
    assert cos_sim > 0.95


def _testing_rebar_model() -> RebarModel:
    return RebarModel(
        encoder=MLPEncoder(
            channels=(10,), n_vocab=8, n_latent=16, d_emb=4, device=torch.device("cpu")
        ),
        decoder=MLPDecoder(
            channels=(10,), n_vocab=8, n_latent=16, d_emb=4, device=torch.device("cpu")
        ),
        init_lam=0.1,
    )
