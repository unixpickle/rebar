import argparse
import json
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from rebar.decoder import MLPDecoder
from rebar.encoder import MLPEncoder
from rebar.rebar import gumbel, hard_threshold
from rebar.rebar_model import RebarModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
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
    parser.add_argument("--samples_path", default=None, type=str)
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
        kls = []
        variances = []
        for xs, _ys in tqdm(train_loader):
            batch = xs.flatten(1).float().to(device)
            opt.zero_grad()
            loss_dict = model.backward(
                batch.float(),
                lambda x: F.binary_cross_entropy_with_logits(
                    x, batch, reduction="none"
                ).sum(1),
                kl_coeff=1.0,
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
            kls.extend(loss_dict["kl"].tolist())
            variances.append(variance.item())
            train_log["variance"].append(variance.item())
            train_log["kl"].append(loss_dict["kl"].mean().item())
            train_log["loss"].append(loss_dict["loss"].mean().item())
            train_log["vlb"].append(train_log["kl"][-1] + train_log["loss"][-1])
        variance = np.mean(variances)
        kl = np.mean(kls)
        loss = np.mean(losses)
        vlb = loss + kl
        print(f"{epoch=} {variance=} {kl=} {loss=} {vlb=}")

    if args.log_path:
        np.savez(args.log_path, train_args=json.dumps(args.__dict__), **train_log)

    if args.samples_path:
        n_grid = 8
        dist = torch.distributions.Categorical(
            logits=torch.zeros(n_grid**2, args.n_latent, args.n_vocab, device=device)
        )
        samples = model.decoder(
            hard_threshold(gumbel(torch.rand_like(dist.logits), dist.logits))
        )
        images = (
            (
                samples.reshape(n_grid, n_grid, 28, 28, 1)
                .sigmoid()
                .permute(0, 2, 1, 3, 4)
                .reshape(28 * n_grid, 28 * n_grid, 1)
                * 255
            )
            .to(torch.uint8)
            .cpu()
            .repeat(1, 1, 3)
            .numpy()
        )
        Image.fromarray(images).save(args.samples_path)


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
