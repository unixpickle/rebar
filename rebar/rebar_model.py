import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class RebarModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        self.encoder = encoder
        self.decoder = decoder
