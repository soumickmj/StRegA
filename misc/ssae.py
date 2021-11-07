import os
import cv2
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
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
import torch.distributions as dist
import math
from torch.cuda.amp import autocast, GradScaler

from datasets import MoodTrainSet, MoodValSet
from datasets.ixi import IXITrainSet
from models.ceVae.aes import VAE
from models.ceVae.helpers import kl_loss_fn, rec_loss_fn, geco_beta_update, get_ema, get_square_mask
import torch.nn.functional as F

#Initial params
gpuID="0"
seed = 1701
num_workers=0
batch_size = 16
log_freq = 10
checkpoint2load = False
checkpoint = r'/scratch/sagrawal/cevae_seg_ixi_t1_t2_pd_mood_noaug/cevae_seg_ixi_t1_t2_pd_mood_noaug-epoch-44.pth.tar'
useCuda = True

ixi_t1_indices = list(range(100,350))
ixi_t2_indices = list(range(100,350))
ixi_proton_indices = list(range(100,350))
mood_t1_indices = list(range(100,350))

ixi_t1_eval_indices = list(range(30,50))
ixi_t2_eval_indices = list(range(30,50))
ixi_proton_eval_indices = list(range(30,50))
mood_t1_eval_indices = list(range(30,50))

ixi_t1_train = r"/project/ptummala/ixi/ixi_t1_segmented_581_3D.hdf5"
ixi_t2_train = r"/project/ptummala/ixi/ixi_t2_segmented_578_3D.hdf5"
ixi_proton_train = r"/project/ptummala/ixi/ixi_pd_segmented_578_3D.hdf5"
mood_t1_train = r"/project/ptummala/moods/mood_t1_seg.h5"
preload_h5 = False

save_path = r'/scratch/ptummala/ssvae'

#Training params
trainID="SSCVAE-1" #ceVAE2D_seg_ixi_mood
num_epochs = 250
lr = 1e-4
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
torch.backends.cudnn.benchmark = False

if __name__ == "__main__" :
    wandb.init(project='anomaly', entity='s9chroma')
    wandb.watch(model)
    config = wandb.config
    config.learning_rate = learning_rate
    wandb.run.name = 'SSVAE1'
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and useCuda else "cpu")
    #device = 'cpu'
    
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
    

    if len(input_dim) == 2:
        conv = nn.Conv2d
        convt = nn.ConvTranspose2d
        d = 2
    else:
        conv = nn.Conv3d
        convt = nn.ConvTranspose3d
        d = 3

    model = VAE(input_size=input_size, z_dim=z_dim, fmap_sizes=model_feature_map_sizes,
               conv_op=conv,
               tconv_op=convt,
               activation_op=torch.nn.PReLU)
    model.d = d
    model.to(device)
    wandb.watch(model)
    optimizer = Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    if checkpoint2load:
        chk = torch.load(checkpoint,map_location=device)
        #model = chk['state_dict']
        #optimizer = chk['optimizer']
        #scaler = chk['AMPScaler']
        start_epoch = 45

        model_dict = model.state_dict()
        pretrained_dict = chk['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        model.to(device) 
        scaler.load_state_dict(chk['AMPScaler'])
        optimizer.load_state_dict(chk['optimizer'])
        del chk
    else:
        start_epoch = 0
        best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        runningLoss = 0.0
        runningLossCounter = 0.0
        train_loss = 0.0
        kl_loss_tot = 0.0
        loss_vae_tot = 0.0
        loss_ce_tot = 0.0
        print('Epoch '+ str(epoch)+ ': Train')
        with tqdm(total=len(train_loader)) as pbar:
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
                images = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:])
                
                I1 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                for i, im in enumerate(images):
                    blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                    scale_percent = 50 # percent of original size
                    width = int(blur.shape[1] * scale_percent / 100)
                    height = int(blur.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                    I1[i][0] = torch.tensor(downsampled)
                U1 = torch.zeros_like(images)
                for i, im in enumerate(U1):
                    dim = (shape[2], shape[3])
                    U1[i][0] = torch.tensor(cv2.resize(I1[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                shape = I1.shape
                I2 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                for i, im in enumerate(I1):
                    blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                    scale_percent = 50 # percent of original size
                    width = int(blur.shape[1] * scale_percent / 100)
                    height = int(blur.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                    I2[i][0] = torch.tensor(downsampled)
                U2 = torch.zeros_like(I1)
                for i, im in enumerate(U2):
                    dim = (shape[2], shape[3])
                    U2[i][0] = torch.tensor(cv2.resize(I2[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                shape = I2.shape
                I3 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                for i, im in enumerate(I2):
                    blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                    scale_percent = 50 # percent of original size
                    width = int(blur.shape[1] * scale_percent / 100)
                    height = int(blur.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                    I3[i][0] = torch.tensor(downsampled)
                U3 = torch.zeros_like(I2)
                for i, im in enumerate(U3):
                    dim = (shape[2], shape[3])
                    U3[i][0] = torch.tensor(cv2.resize(I3[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                L1 = images - U1
                L2 = I1 - U2
                L3 = I2 - U3
                
                images = Variable(L1).to(device)
                optimizer.zero_grad()

                ### VAE Part
                with autocast():
                    loss_vae = 0
                    if ce_factor < 1:
                        x_r, z_dist = model(images)

                        kl_loss = 0
                        if model.d == 3:
                            kl_loss = kl_loss_fn(z_dist, sumdim=(1,)) * beta #check TODO
                            rec_loss_vae = rec_loss_fn(x_r, images, sumdim=(1,2,3,4))
                        else:
                            kl_loss = kl_loss_fn(z_dist, sumdim=(1,2,3)) * beta
                            rec_loss_vae = rec_loss_fn(x_r, images, sumdim=(1,2,3))
                        loss_vae = kl_loss + rec_loss_vae * theta

                

                ### CE Part
                loss_ce = 0
                if ce_factor > 0:

                    ce_tensor = get_square_mask(
                        tensor.shape,
                        square_size=(0, np.max(input_size[1:]) // 2),
                        noise_val=(torch.min(tensor).item(), torch.max(tensor).item()),
                        n_squares=(0, 3),
                    )

                    ce_tensor = torch.from_numpy(ce_tensor).float().to(device)
                    inpt_noisy = torch.where(ce_tensor != 0, ce_tensor, tensor)

                    inpt_noisy = inpt_noisy.to(device)

                    with autocast():
                        x_rec_ce, _ = model(inpt_noisy)
                        if model.d == 3:
                            rec_loss_ce = rec_loss_fn(x_rec_ce, images, sumdim=(1,2,3,4))
                        else:
                            rec_loss_ce = rec_loss_fn(x_rec_ce, images, sumdim=(1,2,3))
                        loss_ce = rec_loss_ce
                        loss = (1.0 - ce_factor) * loss_vae + ce_factor * loss_ce

                if use_geco and ce_factor < 1:
                    g_goal = 0.1
                    g_lr = 1e-4
                    vae_loss_ema = (1.0 - 0.9) * rec_loss_vae + 0.9 * vae_loss_ema
                    theta = geco_beta_update(theta, vae_loss_ema, g_goal, g_lr, speedup=2)
                    

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss = round(loss.item(),4)
                train_loss += loss
                runningLoss += loss
                kl_loss_tot += round(kl_loss.item(),4)
                loss_vae_tot += round(loss_vae.item(),4)
                loss_ce_tot += round(loss_ce.item(),4)
                runningLossCounter += 1
                print("Epoch: ", epoch, ", i: ", i, ", Training loss: ", loss)
                logging.info('[%d/%d][%d/%d] Train Loss: %.4f' % ((epoch+1), num_epochs, i, len(train_loader), loss))
                #For tensorboard
                if i % log_freq == 0:
                    niter = epoch*len(train_loader)+i
                    tb_writer.add_scalar('Train/Loss', runningLoss/runningLossCounter, niter)
                    runningLoss = 0.0
                    runningLossCounter = 0.0
                pbar.update(1)

        wandb.log({"train_loss": train_loss})
        wandb.log({"kl_loss": kl_loss_tot})
        wandb.log({"vae_loss": loss_vae_tot})
        wandb.log({"ce_loss": loss_ce_tot})

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'AMPScaler': scaler.state_dict()
        }
        if epoch%4==0:
            torch.save(checkpoint, os.path.join(save_path, trainID + '-epoch-' + str(epoch) + ".pth.tar"))
            #torch.save(checkpoint, os.path.join(save_path, trainID+".pth.tar"))
        tb_writer.add_scalar('Train/AvgLossEpoch', train_loss/len(train_loader), epoch)



        model.eval()
        mood_val_loss = 0
        with torch.no_grad():
            print('Epoch '+ str(epoch)+ ': Ixi Val')
            with tqdm(total=len(ixi_eval_loader)) as pbar:
                for i, data in enumerate(ixi_eval_loader):
                    img = data['img']['data'].squeeze(-1)
                    tmp = img.view(img.shape[0], 1, -1)
                    min_vals = tmp.min(2, keepdim=True).values
                    max_vals = tmp.max(2, keepdim=True).values
                    tmp = (tmp - min_vals) / max_vals
                    x = tmp.view(img.size())
                    
                    shape = x.shape
                    tensor_reshaped = x.reshape(shape[0],-1)
                    tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
                    images = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:])
                    
                    I1 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                    for i, im in enumerate(images):
                        blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                        scale_percent = 50 # percent of original size
                        width = int(blur.shape[1] * scale_percent / 100)
                        height = int(blur.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                        I1[i][0] = torch.tensor(downsampled)
                    U1 = torch.zeros_like(images)
                    for i, im in enumerate(U1):
                        dim = (shape[2], shape[3])
                        U1[i][0] = torch.tensor(cv2.resize(I1[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                    shape = I1.shape
                    I2 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                    for i, im in enumerate(I1):
                        blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                        scale_percent = 50 # percent of original size
                        width = int(blur.shape[1] * scale_percent / 100)
                        height = int(blur.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                        I2[i][0] = torch.tensor(downsampled)
                    U2 = torch.zeros_like(I1)
                    for i, im in enumerate(U2):
                        dim = (shape[2], shape[3])
                        U2[i][0] = torch.tensor(cv2.resize(I2[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                    shape = I2.shape
                    I3 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                    for i, im in enumerate(I2):
                        blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                        scale_percent = 50 # percent of original size
                        width = int(blur.shape[1] * scale_percent / 100)
                        height = int(blur.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                        I3[i][0] = torch.tensor(downsampled)
                    U3 = torch.zeros_like(I2)
                    for i, im in enumerate(U3):
                        dim = (shape[2], shape[3])
                        U3[i][0] = torch.tensor(cv2.resize(I3[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                    L1 = images - U1
                    L2 = I1 - U2
                    L3 = I2 - U3

                    images = Variable(L1).to(device)
                    
                    x_r, z_dist = model(images)
                    kl_loss = 0
                    kl_loss = kl_loss_fn(z_dist, sumdim=(1,2,3)) * beta
                    rec_loss_vae = rec_loss_fn(x_r, images, sumdim=(1,2,3))
                    loss_vae = kl_loss + rec_loss_vae * theta
                    mood_val_loss += loss_vae.item()
                    pbar.update(1)
                wandb.log({"Val_loss_mood": mood_val_loss})


        #model.eval()
        val_loss = 0
        with torch.no_grad():
            print('Epoch '+ str(epoch)+ ': Mood Val')
            with tqdm(total=len(mood_eval_loader)) as pbar:
                for i, data in enumerate(mood_eval_loader):
                    img = data['img']['data'].squeeze(-1)
                    tmp = img.view(img.shape[0], 1, -1)
                    min_vals = tmp.min(2, keepdim=True).values
                    max_vals = tmp.max(2, keepdim=True).values
                    tmp = (tmp - min_vals) / max_vals
                    x = tmp.view(img.size())
                    
                    shape = x.shape
                    tensor_reshaped = x.reshape(shape[0],-1)
                    tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
                    images = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:])
                    
                    I1 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                    for i, im in enumerate(images):
                        blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                        scale_percent = 50 # percent of original size
                        width = int(blur.shape[1] * scale_percent / 100)
                        height = int(blur.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                        I1[i][0] = torch.tensor(downsampled)
                    U1 = torch.zeros_like(images)
                    for i, im in enumerate(U1):
                        dim = (shape[2], shape[3])
                        U1[i][0] = torch.tensor(cv2.resize(I1[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                    shape = I1.shape
                    I2 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                    for i, im in enumerate(I1):
                        blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                        scale_percent = 50 # percent of original size
                        width = int(blur.shape[1] * scale_percent / 100)
                        height = int(blur.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                        I2[i][0] = torch.tensor(downsampled)
                    U2 = torch.zeros_like(I1)
                    for i, im in enumerate(U2):
                        dim = (shape[2], shape[3])
                        U2[i][0] = torch.tensor(cv2.resize(I2[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                    shape = I2.shape
                    I3 = torch.zeros(shape[0], shape[1], shape[2]//2, shape[3]//2)
                    for i, im in enumerate(I2):
                        blur = cv2.GaussianBlur(im[0].numpy(), (5, 5), 5)
                        scale_percent = 50 # percent of original size
                        width = int(blur.shape[1] * scale_percent / 100)
                        height = int(blur.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        downsampled = cv2.resize(blur, dim, interpolation = cv2.INTER_AREA)
                        I3[i][0] = torch.tensor(downsampled)
                    U3 = torch.zeros_like(I2)
                    for i, im in enumerate(U3):
                        dim = (shape[2], shape[3])
                        U3[i][0] = torch.tensor(cv2.resize(I3[i][0].numpy(), dim, interpolation = cv2.INTER_LINEAR))

                    L1 = images - U1
                    L2 = I1 - U2
                    L3 = I2 - U3

                    images = Variable(L1).to(device)

                    x_r, z_dist = model(images)
                    kl_loss = 0
                    kl_loss = kl_loss_fn(z_dist, sumdim=(1,2,3)) * beta
                    rec_loss_vae = rec_loss_fn(x_r, images, sumdim=(1,2,3))
                    loss_vae = kl_loss + rec_loss_vae * theta
                    val_loss += loss_vae.item()
                    pbar.update(1)
                wandb.log({"Val_loss_oasis": val_loss})