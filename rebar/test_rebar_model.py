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
    actual = model.variance(batch, loss_fn).detach()

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


def _testing_rebar_model() -> RebarModel:
    return RebarModel(
        encoder=MLPEncoder(
            channels=(10,), n_vocab=8, n_latent=16, d_emb=4, device=torch.device("cpu")
        ),
        decoder=MLPDecoder(
            channels=(10,), n_vocab=8, n_latent=16, d_emb=4, device=torch.device("cpu")
        ),
    )
