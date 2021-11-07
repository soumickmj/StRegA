import collections
import random
import re
import os
from typing import List
from tqdm import tqdm
import numpy as np
import torch
import torch.distributions as dist
#from trixi.util.pytorchutils import get_smooth_image_gradient
from ce_noise import smooth_tensor

def kl_loss_fn(z_post, sum_samples=True, correct=False, sumdim=(1,2,3)):
    z_prior = dist.Normal(0, 1.0)
    kl_div = dist.kl_divergence(z_post, z_prior)
    if correct:
        kl_div = torch.sum(kl_div, dim=sumdim)
    else:
        kl_div = torch.mean(kl_div, dim=sumdim)
    if sum_samples:
        return torch.mean(kl_div)
    else:
        return kl_div

def rec_loss_fn(recon_x, x, sum_samples=True, correct=False, sumdim=(1,2,3)):
    if correct:
        x_dist = dist.Laplace(recon_x, 1.0)
        log_p_x_z = x_dist.log_prob(x)
        log_p_x_z = torch.sum(log_p_x_z, dim=sumdim)
    else:
        log_p_x_z = -torch.abs(recon_x - x)
        log_p_x_z = torch.mean(log_p_x_z, dim=sumdim)
    if sum_samples:
        return -torch.mean(log_p_x_z)
    else:
        return -log_p_x_z

def geco_beta_update(beta, error_ema, goal, step_size, min_clamp=1e-10, max_clamp=1e4, speedup=None):
    constraint = (error_ema - goal).detach()
    if speedup is not None and constraint > 0.0:
        beta = beta * torch.exp(speedup * step_size * constraint)
    else:
        beta = beta * torch.exp(step_size * constraint)
    if min_clamp is not None:
        beta = np.max((beta.item(), min_clamp))
    if max_clamp is not None:
        beta = np.min((beta.item(), max_clamp))
    return beta

def get_ema(new, old, alpha):
    if old is None:
        return new
    return (1.0 - alpha) * new + alpha * old


def get_range_val(value, rnd_type="uniform"):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 2:
            if value[0] == value[1]:
                n_val = value[0]
            else:
                orig_type = type(value[0])
                if rnd_type == "uniform":
                    n_val = random.uniform(value[0], value[1])
                elif rnd_type == "normal":
                    n_val = random.normalvariate(value[0], value[1])
                n_val = orig_type(n_val)
        elif len(value) == 1:
            n_val = value[0]
        else:
            raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
        return n_val
    else:
        return value
    
def get_square_mask(data_shape, square_size, n_squares, noise_val=(0, 0), channel_wise_n_val=False, square_pos=None):
    """Returns a 'mask' with the same size as the data, where random squares are != 0

    Args:
        data_shape ([tensor]): [data_shape to determine the shape of the returned tensor]
        square_size ([tuple]): [int/ int tuple (min_size, max_size), determining the min and max squear size]
        n_squares ([type]): [int/ int tuple (min_number, max_number), determining the min and max number of squares]
        noise_val (tuple, optional): [int/ int tuple (min_val, max_val), determining the min and max value given in the 
                                        squares, which habe the value != 0 ]. Defaults to (0, 0).
        channel_wise_n_val (bool, optional): [Use a different value for each channel]. Defaults to False.
        square_pos ([type], optional): [Square position]. Defaults to None.
    """

    def mask_random_square(img_shape, square_size, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks (sets = 0) a random square in an image"""

        img_h = img_shape[-2]
        img_w = img_shape[-1]

        img = np.zeros(img_shape)

        if square_pos is None:
            w_start = np.random.randint(0, img_w - square_size)
            h_start = np.random.randint(0, img_h - square_size)
        else:
            pos_wh = square_pos[np.random.randint(0, len(square_pos))]
            w_start = pos_wh[0]
            h_start = pos_wh[1]

        if img.ndim == 2:
            rnd_n_val = get_range_val(n_val)
            img[h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
        elif img.ndim == 3:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[i, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
        elif img.ndim == 4:
            if channel_wise_n_val:
                for i in range(img.shape[0]):
                    rnd_n_val = get_range_val(n_val)
                    img[:, i, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val
            else:
                rnd_n_val = get_range_val(n_val)
                img[:, :, h_start : (h_start + square_size), w_start : (w_start + square_size)] = rnd_n_val

        return img

    def mask_random_squares(img_shape, square_size, n_squares, n_val, channel_wise_n_val=False, square_pos=None):
        """Masks a given number of squares in an image"""
        img = np.zeros(img_shape)
        for i in range(n_squares):
            img = mask_random_square(
                img_shape, square_size, n_val, channel_wise_n_val=channel_wise_n_val, square_pos=square_pos
            )
        return img

    ret_data = np.zeros(data_shape)
    for sample_idx in range(data_shape[0]):
        # rnd_n_val = get_range_val(noise_val)
        rnd_square_size = get_range_val(square_size)
        rnd_n_squares = get_range_val(n_squares)

        ret_data[sample_idx] = mask_random_squares(
            data_shape[1:],
            square_size=rnd_square_size,
            n_squares=rnd_n_squares,
            n_val=noise_val,
            channel_wise_n_val=channel_wise_n_val,
            square_pos=square_pos,
        )

    return ret_data


