import os
import sys
import wandb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchio
import torchio as tio
from torch.optim import Adam
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from helpers import kl_loss_fn, rec_loss_fn, geco_beta_update, get_ema, get_square_mask

def train(model, train_loader, validation_loader, num_epochs, optimizer, scaler, device):
    for epoch in range(num_epochs):
        model.train()
        print('Epoch ' + str(epoch) + ': Train')
        for i, data in enumerate(train_loader):
            img = data
            
            tmp = img.view(img.shape[0], 1, -1)
            min_vals = tmp.min(2, keepdim=True).values
            max_vals = tmp.max(2, keepdim=True).values
            tmp = (tmp - min_vals) / max_vals
            x = tmp.view(img.size())

            shape = x.shape
            tensor_reshaped = x.reshape(shape[0],-1)
            tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
            tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:]).to(device)

            images = Variable(tensor).to(device) 
            optimizer.zero_grad()
            
            ### VAE Part
            with autocast():
                loss_vae = 0
                if ce_factor < 1:
                    x_r, z_dist = model(images)

                    kl_loss = 0
                    if model.d == 3:
                        kl_loss = kl_loss_fn(z_dist, sumdim=(1,)) * beta
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
            print("Epoch: ", epoch, ", i: ", i, ", Training loss: ", loss)

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'AMPScaler': scaler.state_dict()
        }
        if epoch%4==0:
            torch.save(checkpoint, os.path.join(save_path, trainID + '-epoch-' + str(epoch) + ".pth.tar"))
            
        model.eval()
        with torch.no_grad():
            print('Epoch '+ str(epoch)+ ': Val')
            for i, data in enumerate(validation_loader):
                img = data
                tmp = img.view(img.shape[0], 1, -1)
                min_vals = tmp.min(2, keepdim=True).values
                max_vals = tmp.max(2, keepdim=True).values
                tmp = (tmp - min_vals) / max_vals
                x = tmp.view(img.size())

                shape = x.shape
                tensor_reshaped = x.reshape(shape[0],-1)
                tensor_reshaped = tensor_reshaped[~torch.any(tensor_reshaped.isnan(),dim=1)]
                tensor = tensor_reshaped.reshape(tensor_reshaped.shape[0],*shape[1:])

                images = Variable(tensor).to(device) 
                x_r, z_dist = model(images)
                kl_loss = 0
                kl_loss = kl_loss_fn(z_dist, sumdim=(1,2,3)) * beta
                rec_loss_vae = rec_loss_fn(x_r, images, sumdim=(1,2,3))
                loss_vae = kl_loss + rec_loss_vae * theta
                mood_val_loss += loss_vae.item()
                print("Validation loss: " + str(mood_val_loss))

    return model