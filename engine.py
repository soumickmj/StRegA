from model import VAE

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim = (256, 256) # Slice shapes
input_size = (1,256,256)
z_dim = 1024
model_feature_map_sizes=(16, 64, 256, 1024) # Compact vae

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

lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
scaler = GradScaler()

# Call train function in train.py with created dataloaders
