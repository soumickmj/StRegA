import os
import pickle
import json
import random
import logging
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchio
import torchio as tio
from tqdm import tqdm
import sys
import wandb
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.distributions as dist
import math
from torch.cuda.amp import autocast, GradScaler

from datasets import MoodTrainSet, MoodValSet
from datasets.ixi import IXITrainSet
import torch.nn.functional as F

#Initial params
gpuID="0"
seed = 1701
num_workers=0
batch_size = 16
log_freq = 10
useCuda = True

ixi_t1_indices = list(range(100,350))
ixi_t2_indices = list(range(100,350))
ixi_proton_indices = list(range(100,350))
mood_t1_indices = list(range(100,350))

ixi_t1_eval_indices = list(range(30,50))
ixi_t2_eval_indices = list(range(30,50))
ixi_proton_eval_indices = list(range(30,50))
mood_t1_eval_indices = list(range(30,50))

ixi_t1_train = r"/nfs1/sagrawal/IXI/ixi_t1_segmented_581_3D.hdf5"
ixi_t2_train = r"/nfs1/sagrawal/IXI/ixi_t2_segmented_578_3D.hdf5"
ixi_proton_train = r"/nfs1/sagrawal/IXI/ixi_pd_segmented_578_3D.hdf5"
mood_t1_train = r"/nfs1/sagrawal/Data/Project_Anomaly/mood_t1_seg.h5"
preload_h5 = False

log_path = r'/scratch/ptummala/Logs/gmvae'
save_path = r'/scratch/ptummala/Saved/gmvae'

#Training params
trainID="GMVAE"
num_epochs = 100
learning_rate = 5e-4
patch_size=(256,256,1) #Set it to None if not desired
patchQ_len = 512
patches_per_volume = 256  
log_freq = 10 
preload_h5 = False

#Network Params
IsVAE=True
input_shape=(256,256,256)
input_dim = (256,256)
input_size = (1,256,256)
z_dim=1024
model_feature_map_sizes=(16, 64, 256, 1024)
n_channels=1
ce_factor=0.5
beta=0.01
vae_loss_ema = 1
theta = 1
use_geco=False

os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() and useCuda else "cpu")
#device = 'cpu'

def conv_block(in_channels, out_channels, kernel_size, padding, stride):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        ),
        nn.ReLU(inplace=True),
    )

def transp_conv_block(in_channels, out_channels, kernel_size, padding, stride, output_padding = 0):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            output_padding=output_padding,
        ),
        nn.ReLU(inplace=True),
    )

class GMVAE(nn.Module):
    def __init__(self):
        super(GMVAE, self).__init__()

        self.dim_c = 9
        self.dim_z = 1
        self.dim_w = 1

        self.conv_block1 = conv_block(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1, stride = 2)
        self.conv_block2 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.conv_block3 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.conv_block4 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 2)
        self.conv_block5 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.conv_block6 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)

        self.w_mu_layer = nn.Conv2d(64, self.dim_w, kernel_size = 1, padding = 0, stride = 1)
        self.w_log_sigma_layer = nn.Conv2d(64, self.dim_w, kernel_size = 1, padding = 0, stride = 1)

        self.z_mu_layer = nn.Conv2d(64, self.dim_z, kernel_size = 1, padding = 0, stride = 1)
        self.z_log_sigma_layer = nn.Conv2d(64, self.dim_z, kernel_size = 1, padding = 0, stride = 1)

        self.conv_block7 = conv_block(in_channels = 1, out_channels = 64, kernel_size = 1, padding = 0, stride = 1)

        self.z_wc_mu_layer = nn.Conv2d(64, self.dim_z * self.dim_c, kernel_size = 1, padding = 0, stride = 1)
        self.z_wc_log_sigma_layer = nn.Conv2d(64, self.dim_z * self.dim_c, kernel_size = 1, padding = 0, stride = 1)

        self.conv_block8 = conv_block(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.transp_conv_block1 = transp_conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.transp_conv_block2 = transp_conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)

        self.conv_block9 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.transp_conv_block3 = transp_conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.transp_conv_block4 = transp_conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)

        self.conv_block10 = conv_block(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)

        self.xz_mu_layer = nn.Conv2d(64, 1, kernel_size = 3, padding = 1, stride = 1)

    def forward(self, image):
        outputs = {}

        # encoding network q(z|x) and q(w|x)
        x1 = self.conv_block1(image)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)
        x4 = self.conv_block4(x3)
        x5 = self.conv_block5(x4)
        x6 = self.conv_block6(x5)

        outputs['w_mu'] = w_mu = self.w_mu_layer(x6)
        outputs['w_log_sigma'] = w_log_sigma = self.w_log_sigma_layer(x6)
        # reparametrization
        rand_w = (torch.randn(w_log_sigma.shape).to(device) * torch.exp(0.5 * w_log_sigma).to(device)).to(device)
        outputs['w_sampled'] = w_sampled = w_mu + rand_w

        outputs['z_mu'] = z_mu = self.z_mu_layer(x6)
        outputs['z_log_sigma'] = z_log_sigma = self.z_log_sigma_layer(x6)
        # reparametrization
        rand_z = (torch.randn(z_log_sigma.shape).to(device) * torch.exp(0.5 * z_log_sigma).to(device)).to(device)
        outputs['z_sampled'] = z_sampled = z_mu + rand_z

        # posterior p(z|w,c)
        x7 = self.conv_block7(w_sampled)
        z_wc_mu = self.z_wc_mu_layer(x7)
        z_wc_log_sigma = self.z_wc_log_sigma_layer(x7)

        bias = torch.full((z_wc_log_sigma.shape[1],), 0.1).view(1, -1, 1, 1).to(device)
        z_wc_log_sigma_inv = z_wc_log_sigma + bias

        outputs['z_wc_mus'] = z_wc_mus = z_wc_mu.view(-1, self.dim_c, self.dim_z, z_wc_mu.shape[2], z_wc_mu.shape[3])
        outputs['z_wc_log_sigma_invs'] = z_wc_log_sigma_invs = z_wc_log_sigma_inv.view(-1, self.dim_c, self.dim_z, z_wc_log_sigma_inv.shape[2], z_wc_log_sigma_inv.shape[3])
        # reparametrization
        rand_z_wc = (torch.randn(z_wc_log_sigma_invs.shape).to(device) * torch.exp(z_wc_log_sigma_invs).to(device)).to(device)
        outputs['z_wc_sampled'] = z_wc_sampled = z_wc_mus + rand_z_wc

        # decoder p(x|z)
        x8 = self.conv_block8(z_sampled)
        x9 = self.transp_conv_block1(x8)
        x10 = dec_part1 = self.transp_conv_block2(x9)

        dec_part1_reshaped = nn.Upsample(scale_factor=2, mode='nearest')(dec_part1)
        dec_part1_padded = dec_part1_reshaped

        x11 = self.conv_block9(dec_part1_padded)
        x12 = self.transp_conv_block3(x11)
        x13 = dec_part2 = self.transp_conv_block4(x12)

        dec_part2_reshaped = nn.Upsample(scale_factor=2, mode='nearest')(dec_part2)
        dec_part2_padded = dec_part2_reshaped

        dec = self.conv_block10(dec_part2_padded)
        outputs['xz_mu'] = xz_mu = self.xz_mu_layer(dec)

        # prior p(c)
        z_sample = z_sampled.unsqueeze(dim = 1).repeat(1, self.dim_c, 1, 1, 1)
        loglh = -0.5 * (((z_sample - z_wc_mus) ** 2) * torch.exp(z_wc_log_sigma_invs)) - z_wc_log_sigma_invs + torch.log(torch.tensor(np.pi))
        loglh_sum = torch.sum(loglh, dim = 2)
        outputs['pc_logit'] = pc_logit = loglh_sum
        outputs['pc'] = pc = nn.Softmax(dim = 1)(loglh_sum)

        return outputs

class GMVAE_Trainer():
    def __init__(self):
        super(GMVAE_Trainer, self).__init__()
        self.model = GMVAE()
        self.model.to(device)

        self.optimizer = Adam(self.model.parameters(), lr = learning_rate)
        self.scaler = GradScaler()
        self.dim_c = 9
        self.dim_z = 1
        self.dim_w = 1
        self.c_lambda = 0.5
        self.restore_lr = 1e-3
        self.restore_steps = 0
        self.tv_lambda = -1.0

        wandb.init(project='anomaly', entity='s9chroma')
        wandb.watch(self.model)
        config = wandb.config
        config.learning_rate = learning_rate
        wandb.run.name = 'GMVAE'
        
    def total_variation(self, images):
        # The input is a batch of images with shape:
        # [batch, channels, height, width].

        # Calculate the difference of neighboring pixel-values.
        # The images are shifted one pixel along the height and width by slicing.
        pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
        pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]

        return torch.abs(pixel_dif1).sum(dim=1).sum(dim=1).sum(dim=1) + torch.abs(
            pixel_dif2
        ).sum(dim=1).sum(dim=1).sum(dim=1)
    
    def train(self, train_loader, ixi_val_loader, mood_val_loader):
        for epoch in range(num_epochs):
            print("")
            self.model.train()
            train_total_loss = 0.0
            train_mean_p_loss = 0.0
            train_con_loss = 0.0
            train_mean_w_loss = 0.0
            train_mean_c_loss = 0.0
            print('Epoch '+ str(epoch)+ ': Training')
            for i, data in enumerate(train_loader):
                img = data['img']['data'].squeeze(-1)

                tmp = img.view(img.shape[0], 1, -1)
                min_vals = tmp.min(2, keepdim=True).values
                max_vals = tmp.max(2, keepdim=True).values
                tmp = (tmp - min_vals) / max_vals
                x = tmp.view(img.size())

                shape = x.shape
                tensor_reshaped = x.reshape(shape[0],-1)
                tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
                tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:]).to(device)

                images = Variable(tensor, requires_grad = True).to(device)
                self.optimizer.zero_grad()
                
                #with autocast():
                outputs = self.model(images)

                w_mu = outputs['w_mu']
                w_log_sigma = outputs['w_log_sigma']
                z_sampled = outputs['z_sampled']
                z_mu = outputs['z_mu']
                z_log_sigma = outputs['z_log_sigma']
                z_wc_mu = outputs['z_wc_mus']
                z_wc_log_sigma_inv = outputs['z_wc_log_sigma_invs']
                xz_mu = outputs['xz_mu']
                pc = outputs['pc']
                reconstruction = xz_mu

                # Build Losses
                # 1) Reconstruction Loss
                mean_p_loss = nn.L1Loss()(images, xz_mu)

                # 2. E_c_w[KL(q(z|x)|| p(z|w, c))]
                # calculate KL for each cluster
                # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
                # then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
                z_mu_u = z_mu.unsqueeze(dim = 1).repeat(1, self.dim_c, 1, 1, 1)
                z_logvar = z_log_sigma.unsqueeze(dim = 1).repeat(1, self.dim_c, 1, 1, 1)
                d_mu_2 = (z_mu_u - z_wc_mu) ** 2
                d_var = (torch.exp(z_logvar) + d_mu_2) * (torch.exp(z_wc_log_sigma_inv) + 1e-6)
                d_logvar = -1 * (z_wc_log_sigma_inv + z_logvar)
                kl = (d_var + d_logvar - 1) * 0.5
                con_prior_loss = torch.sum(torch.matmul(kl, pc.unsqueeze(dim = 2)).squeeze(dim = 2).view(images.shape[0], -1), dim = 1)
                mean_con_loss = torch.mean(con_prior_loss)

                # 3. KL(q(w|x)|| p(w) ~ N(0, I))
                # KL = 1/2 sum( mu^2 + var - logvar -1 )
                w_loss = 0.5 * torch.sum((torch.square(w_mu) + torch.exp(w_log_sigma) - w_log_sigma - 1).view(images.shape[0], -1), dim = 1)
                mean_w_loss = torch.mean(w_loss)

                # 4. KL(q(c|z)||p(c)) =  - sum_k q(k) log p(k)/q(k) , k = dim_c
                # let p(k) = 1/K#
                closs1 = torch.mul(pc, torch.log(pc * self.dim_c + 1e-8)).sum(dim = 1)
                c_lambda = torch.full(closs1.shape, self.c_lambda).to(device)
                c_loss = torch.maximum(closs1, c_lambda)
                c_loss = c_loss.sum(dim = 1).sum(dim = 1)
                mean_c_loss = torch.mean(c_loss)

                loss = mean_p_loss + mean_con_loss + mean_w_loss + mean_c_loss

                train_total_loss += loss
                train_mean_p_loss += mean_p_loss
                train_con_loss += mean_con_loss
                train_mean_w_loss += mean_w_loss
                train_mean_c_loss += mean_c_loss
                
#                 self.scaler.scale(loss).backward()
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
                loss.backward()
                self.optimizer.step()
                print("Training step: ", i)
                print("Total loss: ", loss)
                print("Mean p loss: ", mean_p_loss)
                print("Mean Con Loss: ", mean_con_loss)
                print("Mean w loss: ", mean_w_loss)
                print("Mean c loss: ", mean_c_loss)
                print("")

            wandb.log({"Training Loss(Total)": train_total_loss}, step = epoch)
            wandb.log({"Training Loss(mean_p_loss)": train_mean_p_loss}, step = epoch)
            wandb.log({"Training Loss(mean_con_loss)": train_con_loss}, step = epoch)
            wandb.log({"Training Loss(mean_w_loss)": train_mean_w_loss}, step = epoch)
            wandb.log({"Training Loss(mean_c_loss)": train_mean_c_loss}, step = epoch)
            
            self.model.eval()
            with torch.no_grad():
                valid_ixi_total_loss = 0.0
                valid_ixi_mean_p_loss = 0.0
                valid_ixi_con_loss = 0.0
                valid_ixi_mean_w_loss = 0.0
                valid_ixi_mean_c_loss = 0.0
                print('Epoch '+ str(epoch)+ ': Ixi Validation')
                for i, data in enumerate(ixi_val_loader):
                    img = data['img']['data'].squeeze(-1)

                    tmp = img.view(img.shape[0], 1, -1)
                    min_vals = tmp.min(2, keepdim=True).values
                    max_vals = tmp.max(2, keepdim=True).values
                    tmp = (tmp - min_vals) / max_vals
                    x = tmp.view(img.size())

                    shape = x.shape
                    tensor_reshaped = x.reshape(shape[0],-1)
                    tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
                    tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:]).to(device)

                    images = Variable(tensor, requires_grad = True).to(device)
                    outputs = self.model(images)

                    w_mu = outputs['w_mu']
                    w_log_sigma = outputs['w_log_sigma']
                    z_sampled = outputs['z_sampled']
                    z_mu = outputs['z_mu']
                    z_log_sigma = outputs['z_log_sigma']
                    z_wc_mu = outputs['z_wc_mus']
                    z_wc_log_sigma_inv = outputs['z_wc_log_sigma_invs']
                    xz_mu = outputs['xz_mu']
                    pc = outputs['pc']
                    reconstruction = xz_mu

                    # Build Losses
                    # 1) Reconstruction Loss
                    mean_p_loss = nn.L1Loss()(images, xz_mu)

                    # 2. E_c_w[KL(q(z|x)|| p(z|w, c))]
                    # calculate KL for each cluster
                    # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
                    # then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
                    z_mu_u = z_mu.unsqueeze(dim = 1).repeat(1, self.dim_c, 1, 1, 1)
                    z_logvar = z_log_sigma.unsqueeze(dim = 1).repeat(1, self.dim_c, 1, 1, 1)
                    d_mu_2 = (z_mu_u - z_wc_mu) ** 2
                    d_var = (torch.exp(z_logvar) + d_mu_2) * (torch.exp(z_wc_log_sigma_inv) + 1e-6)
                    d_logvar = -1 * (z_wc_log_sigma_inv + z_logvar)
                    kl = (d_var + d_logvar - 1) * 0.5
                    con_prior_loss = torch.sum(torch.matmul(kl, pc.unsqueeze(dim = 2)).squeeze(dim = 2).view(images.shape[0], -1), dim = 1)
                    mean_con_loss = torch.mean(con_prior_loss)

                    # 3. KL(q(w|x)|| p(w) ~ N(0, I))
                    # KL = 1/2 sum( mu^2 + var - logvar -1 )
                    w_loss = 0.5 * torch.sum((torch.square(w_mu) + torch.exp(w_log_sigma) - w_log_sigma - 1).view(images.shape[0], -1), dim = 1)
                    mean_w_loss = torch.mean(w_loss)

                    # 4. KL(q(c|z)||p(c)) =  - sum_k q(k) log p(k)/q(k) , k = dim_c
                    # let p(k) = 1/K#
                    closs1 = torch.mul(pc, torch.log(pc * self.dim_c + 1e-8)).sum(dim = 1)
                    c_lambda = torch.full(closs1.shape, self.c_lambda).to(device)
                    c_loss = torch.maximum(closs1, c_lambda)
                    c_loss = c_loss.sum(dim = 1).sum(dim = 1)
                    mean_c_loss = torch.mean(c_loss)

                    loss = mean_p_loss + mean_con_loss + mean_w_loss + mean_c_loss
                    
                    valid_ixi_total_loss += loss
                    valid_ixi_mean_p_loss += mean_p_loss
                    valid_ixi_con_loss += mean_con_loss
                    valid_ixi_mean_w_loss += mean_w_loss
                    valid_ixi_mean_c_loss += mean_c_loss
    
                    print("IXI Validation step: ", i)
                    print("Total loss: ", loss)
                    print("Mean p loss: ", mean_p_loss)
                    print("Mean Con Loss: ", mean_con_loss)
                    print("Mean w loss: ", mean_w_loss)
                    print("Mean c loss: ", mean_c_loss)
                    print("")

                wandb.log({"IXI Validation Loss(Total)": valid_ixi_total_loss}, step = epoch)
                wandb.log({"IXI Validation Loss(mean_p_loss)": valid_ixi_mean_p_loss}, step = epoch)
                wandb.log({"IXI Validation Loss(mean_con_loss)": valid_ixi_con_loss}, step = epoch)
                wandb.log({"IXI Validation Loss(mean_w_loss)": valid_ixi_mean_w_loss}, step = epoch)
                wandb.log({"IXI Validation Loss(mean_c_loss)": valid_ixi_mean_c_loss}, step = epoch)
                    
            with torch.no_grad():
                valid_mood_total_loss = 0.0
                valid_mood_mean_p_loss = 0.0
                valid_mood_con_loss = 0.0
                valid_mood_mean_w_loss = 0.0
                valid_mood_mean_c_loss = 0.0
                print('Epoch '+ str(epoch)+ ': MOOD Validation')
                for i, data in enumerate(mood_val_loader):
                    img = data['img']['data'].squeeze(-1)

                    tmp = img.view(img.shape[0], 1, -1)
                    min_vals = tmp.min(2, keepdim=True).values
                    max_vals = tmp.max(2, keepdim=True).values
                    tmp = (tmp - min_vals) / max_vals
                    x = tmp.view(img.size())

                    shape = x.shape
                    tensor_reshaped = x.reshape(shape[0],-1)
                    tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
                    tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:]).to(device)

                    images = Variable(tensor, requires_grad = True).to(device)
                    outputs = self.model(images)

                    w_mu = outputs['w_mu']
                    w_log_sigma = outputs['w_log_sigma']
                    z_sampled = outputs['z_sampled']
                    z_mu = outputs['z_mu']
                    z_log_sigma = outputs['z_log_sigma']
                    z_wc_mu = outputs['z_wc_mus']
                    z_wc_log_sigma_inv = outputs['z_wc_log_sigma_invs']
                    xz_mu = outputs['xz_mu']
                    pc = outputs['pc']
                    reconstruction = xz_mu

                    # Build Losses
                    # 1) Reconstruction Loss
                    mean_p_loss = nn.L1Loss()(images, xz_mu)

                    # 2. E_c_w[KL(q(z|x)|| p(z|w, c))]
                    # calculate KL for each cluster
                    # KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
                    # then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
                    z_mu_u = z_mu.unsqueeze(dim = 1).repeat(1, self.dim_c, 1, 1, 1)
                    z_logvar = z_log_sigma.unsqueeze(dim = 1).repeat(1, self.dim_c, 1, 1, 1)
                    d_mu_2 = (z_mu_u - z_wc_mu) ** 2
                    d_var = (torch.exp(z_logvar) + d_mu_2) * (torch.exp(z_wc_log_sigma_inv) + 1e-6)
                    d_logvar = -1 * (z_wc_log_sigma_inv + z_logvar)
                    kl = (d_var + d_logvar - 1) * 0.5
                    con_prior_loss = torch.sum(torch.matmul(kl, pc.unsqueeze(dim = 2)).squeeze(dim = 2).view(images.shape[0], -1), dim = 1)
                    mean_con_loss = torch.mean(con_prior_loss)

                    # 3. KL(q(w|x)|| p(w) ~ N(0, I))
                    # KL = 1/2 sum( mu^2 + var - logvar -1 )
                    w_loss = 0.5 * torch.sum((torch.square(w_mu) + torch.exp(w_log_sigma) - w_log_sigma - 1).view(images.shape[0], -1), dim = 1)
                    mean_w_loss = torch.mean(w_loss)

                    # 4. KL(q(c|z)||p(c)) =  - sum_k q(k) log p(k)/q(k) , k = dim_c
                    # let p(k) = 1/K#
                    closs1 = torch.mul(pc, torch.log(pc * self.dim_c + 1e-8)).sum(dim = 1)
                    c_lambda = torch.full(closs1.shape, self.c_lambda).to(device)
                    c_loss = torch.maximum(closs1, c_lambda)
                    c_loss = c_loss.sum(dim = 1).sum(dim = 1)
                    mean_c_loss = torch.mean(c_loss)

                    loss = mean_p_loss + mean_con_loss + mean_w_loss + mean_c_loss
                    
                    valid_mood_total_loss += loss
                    valid_mood_mean_p_loss += mean_p_loss
                    valid_mood_con_loss += mean_con_loss
                    valid_mood_mean_w_loss += mean_w_loss
                    valid_mood_mean_c_loss += mean_c_loss
    
                    print("MOOD Validation step: ", i)
                    print("Total loss: ", loss)
                    print("Mean p loss: ", mean_p_loss)
                    print("Mean Con Loss: ", mean_con_loss)
                    print("Mean w loss: ", mean_w_loss)
                    print("Mean c loss: ", mean_c_loss)
                    print("")

                wandb.log({"MOOD Validation Loss(Total)": valid_mood_total_loss}, step = epoch)
                wandb.log({"MOOD Validation Loss(mean_p_loss)": valid_mood_mean_p_loss}, step = epoch)
                wandb.log({"MOOD Validation Loss(mean_con_loss)": valid_mood_con_loss}, step = epoch)
                wandb.log({"MOOD Validation Loss(mean_w_loss)": valid_mood_mean_w_loss}, step = epoch)
                wandb.log({"MOOD Validation Loss(mean_c_loss)": valid_mood_mean_c_loss}, step = epoch)
                
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'AMPScaler': self.scaler.state_dict(),
            }
            print(os.path.join(save_path, trainID + '_' + str(epoch) + ".pth.tar"))
            torch.save(checkpoint, os.path.join(save_path, trainID + '_' + str(epoch) + ".pth.tar"))

if __name__ == "__main__":
    ixi_t1_trainset = IXITrainSet(indices=ixi_t1_indices, data_path=ixi_t1_train, lazypatch=True if patch_size else False, preload=preload_h5)
    ixi_t2_trainset = IXITrainSet(indices=ixi_t2_indices, data_path=ixi_t2_train, lazypatch=True if patch_size else False, preload=preload_h5)
    ixi_pd_trainset = IXITrainSet(indices=ixi_proton_indices, data_path=ixi_proton_train, lazypatch=True if patch_size else False, preload=preload_h5)
    mood_trainset = IXITrainSet(indices=mood_t1_indices, data_path=mood_t1_train, lazypatch=True if patch_size else False, preload=preload_h5)

    tot_trainset = ixi_t1_trainset + ixi_t2_trainset + ixi_pd_trainset + mood_trainset

    ixi_t1_eval = IXITrainSet(indices=ixi_t1_eval_indices, data_path=ixi_t1_train, lazypatch=True if patch_size else False, preload=preload_h5)
    ixi_t2_eval = IXITrainSet(indices=ixi_t2_eval_indices, data_path=ixi_t2_train, lazypatch=True if patch_size else False, preload=preload_h5)
    ixi_pd_eval = IXITrainSet(indices=ixi_proton_eval_indices, data_path=ixi_proton_train, lazypatch=True if patch_size else False, preload=preload_h5)
    mood_t1_eval = IXITrainSet(indices=mood_t1_eval_indices, data_path=mood_t1_train, lazypatch=True if patch_size else False, preload=preload_h5)


    ixi_evalset = ixi_t1_eval + ixi_t2_eval + ixi_pd_eval


    if patch_size:
        input_shape = tuple(x for x in patch_size if x!=1)
        trainset = torchio.data.Queue(
                        subjects_dataset = tot_trainset,
                        max_length = patchQ_len,
                        samples_per_volume = patches_per_volume,
                        sampler = torchio.data.UniformSampler(patch_size=patch_size),
                        # num_workers = num_workers
                    )

        ixi_trainset = torchio.data.Queue(
                        subjects_dataset = ixi_evalset,
                        max_length = patchQ_len,
                        samples_per_volume = patches_per_volume,
                        sampler = torchio.data.UniformSampler(patch_size=patch_size),
                        # num_workers = num_workers
                    )

        mood_trainset = torchio.data.Queue(
                        subjects_dataset = mood_t1_eval,
                        max_length = patchQ_len,
                        samples_per_volume = patches_per_volume,
                        sampler = torchio.data.UniformSampler(patch_size=patch_size),
                        # num_workers = num_workers
                    )

    train_loader = DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    ixi_eval_loader = DataLoader(dataset=ixi_trainset,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    mood_eval_loader = DataLoader(dataset=mood_trainset,batch_size=batch_size,shuffle=True, num_workers=num_workers)
    
    trainer = GMVAE_Trainer()
    trainer.train(train_loader, ixi_eval_loader, mood_eval_loader)
