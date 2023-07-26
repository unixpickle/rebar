import argparse

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
    parser.add_argument("--lr", default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_latent", type=int, default=32)
    parser.add_argument("--n_vocab", type=int, default=64)
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
    )

    opt = Adam(model.parameters(), lr=args.lr)
    for epoch in range(10):
        losses = []
        variances = []
        for xs, _ys in tqdm(train_loader):
            batch = xs.flatten(1).to(device)
            opt.zero_grad()
            hard_losses = model.backward(batch, lambda x: (x - batch).pow(2).mean(1))

            variance = model.variance(batch, lambda x: (x - batch).pow(2).mean(1))
            variance.backward()

            opt.step()
            losses.append(hard_losses.tolist())
            variances.append(variance.item())
        variance = np.mean(variances)
        loss = np.mean(losses)
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
