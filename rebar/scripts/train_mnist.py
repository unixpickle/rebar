import argparse
import json
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--variance_batch_size", type=int, default=4)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_vocab", type=int, default=64)
    parser.add_argument("--d_emb", type=int, default=32)
    parser.add_argument("--sample_baseline", action="store_true")
    parser.add_argument("--log_path", default=None, type=str)
    args = parser.parse_args()

    train_loader = data_loader(args.batch_size, train=True)
    # test_loader = data_loader(args.batch_size, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RebarModel(
        encoder=MLPEncoder(
            channels=(28 * 28, 256),
            n_vocab=args.n_vocab,
            n_latent=args.n_latent,
            d_emb=args.d_emb,
            device=device,
        ),
        decoder=MLPDecoder(
            channels=(256, 28 * 28),
            n_vocab=args.n_vocab,
            n_latent=args.n_latent,
            d_emb=args.d_emb,
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
    train_log = defaultdict(list)
    for epoch in range(args.epochs):
        losses = []
        entropies = []
        variances = []
        for xs, _ys in tqdm(train_loader):
            batch = xs.flatten(1).float().to(device)
            opt.zero_grad()
            loss_dict = model.backward(
                batch.float(),
                lambda x: F.binary_cross_entropy_with_logits(
                    x, batch, reduction="none"
                ).sum(1),
                entropy_coeff=-1.0,
            )

            var_batch = batch[torch.randperm(len(batch))[: args.variance_batch_size]]
            variance = model.variance_backward(
                var_batch,
                lambda x: F.binary_cross_entropy_with_logits(
                    x, var_batch, reduction="none"
                ).sum(1),
            )

            for name, p in model.named_parameters():
                if not p.grad.isfinite().all().item():
                    warnings.warn(f"{name} grad is not finite")

            opt.step()
            losses.extend(loss_dict["loss"].tolist())
            entropies.extend(loss_dict["entropy"].tolist())
            variances.append(variance.item())
            train_log["variance"].append(variance.item())
            train_log["entropy"].append(loss_dict["entropy"].mean().item())
            train_log["loss"].append(loss_dict["loss"].mean().item())
        variance = np.mean(variances)
        entropy = np.mean(entropies)
        loss = np.mean(losses)
        print(f"{epoch=} {variance=} {entropy=} {loss=}")

    if args.log_path:
        np.savez(args.log_path, train_args=json.dumps(args.__dict__), **train_log)


def data_loader(bs: int, train: bool) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), Binarize()])
    dataset = datasets.MNIST(
        "../data", train=train, download=train, transform=transform
    )
    return DataLoader(dataset, batch_size=bs)


class Binarize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x > torch.rand_like(x)


if __name__ == "__main__":
    main()
