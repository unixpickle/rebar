import argparse
import warnings
from random import sample

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from rebar.decoder import MLPDecoder
from rebar.encoder import MLPEncoder
from rebar.rebar_model import RebarModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cv_lr", type=float, default=1e-2)
    parser.add_argument("--init_eta", type=float, default=1.0)
    parser.add_argument("--init_lam", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--variance_batch_size", type=int, default=4)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_vocab", type=int, default=64)
    parser.add_argument("--sample_baseline", action="store_true")
    args = parser.parse_args()

    train_loader = data_loader(args.batch_size, train=True)
    # test_loader = data_loader(args.batch_size, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RebarModel(
        encoder=MLPEncoder(
            channels=(28 * 28, 256),
            n_vocab=args.n_vocab,
            n_latent=args.n_latent,
            d_emb=4,
            device=device,
        ),
        decoder=MLPDecoder(
            channels=(256, 28 * 28),
            n_vocab=args.n_vocab,
            n_latent=args.n_latent,
            d_emb=4,
            device=device,
        ),
        init_eta=args.init_eta,
        init_lam=args.init_lam,
        sample_baseline=args.sample_baseline,
    )

    opt = Adam(
        [
            dict(
                params=(
                    list(model.encoder.parameters()) + list(model.decoder.parameters())
                ),
                lr=args.lr,
            ),
            dict(params=[model.eta_arg, model.lam_arg], lr=args.cv_lr),
        ],
    )
    for epoch in range(10):
        losses = []
        variances = []
        for xs, _ys in tqdm(train_loader):
            batch = xs.flatten(1).to(device)
            opt.zero_grad()
            hard_losses = model.backward(batch, lambda x: (x - batch).pow(2).mean(1))

            var_batch = batch[torch.randperm(len(batch))[: args.variance_batch_size]]
            variance = model.variance(
                var_batch, lambda x: (x - var_batch).pow(2).mean(1)
            )
            variance.backward()

            for name, p in model.named_parameters():
                if not p.grad.isfinite().all().item():
                    warnings.warn(f"{name} grad is not finite")

            opt.step()
            losses.append(hard_losses.tolist())
            variances.append(variance.item())
        variance = np.mean(variances)
        loss = np.mean([x for y in losses for x in y])
        print(f"{epoch=} {variance=} {loss=}")


def data_loader(bs: int, train: bool) -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        "../data", train=train, download=train, transform=transform
    )
    return DataLoader(dataset, batch_size=bs)


if __name__ == "__main__":
    main()
