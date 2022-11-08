import numpy as np
import torch
import torch.distributions as dist

from ae_bases import BasicEncoder, BasicGenerator

class VAE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=256,
        fmap_sizes=(16, 64, 256, 1024),
        to_1x1=True,
        conv_op=torch.nn.Conv2d,
        conv_params=None,
        tconv_op=torch.nn.ConvTranspose2d,
        tconv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=torch.nn.LeakyReLU,
        activation_params=None,
        block_op=None,
        block_params=None,
        *args,
        **kwargs
    ):
        super(VAE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)

        self.enc = BasicEncoder(
            input_size=input_size_enc,
            fmap_sizes=fmap_sizes,
            z_dim=z_dim * 2,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )
        self.dec = BasicGenerator(
            input_size=input_size_dec,
            fmap_sizes=fmap_sizes[::-1],
            z_dim=z_dim,
            upsample_op=tconv_op,
            conv_params=tconv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )

        self.hidden_size = self.enc.output_size

    def forward(self, inpt, sample=True, no_dist=False, **kwargs):
        y1 = self.enc(inpt, **kwargs)

        mu, log_std = torch.chunk(y1, 2, dim=1)
        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)
        z_sample = z_dist.rsample() if sample else mu
        x_rec = self.dec(z_sample)

        return x_rec if no_dist else (x_rec, z_dist)

    def encode(self, inpt, **kwargs):
        enc = self.enc(inpt, **kwargs)
        mu, log_std = torch.chunk(enc, 2, dim=1)
        std = torch.exp(log_std)
        return mu, std

    def decode(self, inpt, **kwargs):
        return self.dec(inpt, **kwargs)

class AE(torch.nn.Module):
    def __init__(
        self,
        input_size,
        z_dim=1024,
        fmap_sizes=(16, 64, 256, 1024),
        to_1x1=True,
        conv_op=torch.nn.Conv2d,
        conv_params=None,
        tconv_op=torch.nn.ConvTranspose2d,
        tconv_params=None,
        normalization_op=None,
        normalization_params=None,
        activation_op=torch.nn.LeakyReLU,
        activation_params=None,
        block_op=None,
        block_params=None,
        *args,
        **kwargs
    ):
        super(AE, self).__init__()

        input_size_enc = list(input_size)
        input_size_dec = list(input_size)

        self.enc = BasicEncoder(
            input_size=input_size_enc,
            fmap_sizes=fmap_sizes,
            z_dim=z_dim,
            conv_op=conv_op,
            conv_params=conv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )
        self.dec = BasicGenerator(
            input_size=input_size_dec,
            fmap_sizes=fmap_sizes[::-1],
            z_dim=z_dim,
            upsample_op=tconv_op,
            conv_params=tconv_params,
            normalization_op=normalization_op,
            normalization_params=normalization_params,
            activation_op=activation_op,
            activation_params=activation_params,
            block_op=block_op,
            block_params=block_params,
            to_1x1=to_1x1,
        )

        self.hidden_size = self.enc.output_size

    def forward(self, inpt, **kwargs):

        y1 = self.enc(inpt, **kwargs)

        return self.dec(y1)

    def encode(self, inpt, **kwargs):
        enc = self.enc(inpt, **kwargs)
        return enc

    def decode(self, inpt, **kwargs):
        return self.dec(inpt, **kwargs)