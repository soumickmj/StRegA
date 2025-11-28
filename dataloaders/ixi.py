import h5py as h5
import numpy as np

import torch
from torch.utils.data import Dataset
from torchio.data.subject import Subject
from .ixi_torchiowrap import IXI_H5DSImage

class IXITrainSet(Dataset):
    def __init__(self, indices=None, data_path='Ixi_with_skull.h5', torchiosub=True, lazypatch=True, preload=False):
        # Support both single path and list of paths
        if isinstance(data_path, str):
            data_path = [data_path]
        
        self.h5_files = [h5.File(path, 'r', swmr=True) for path in data_path]
        self.samples = []
        
        for h5_file in self.h5_files:
            if indices:
                self.samples.extend([h5_file[str(i).zfill(5)] for i in indices])
            else:
                self.samples.extend([h5_file[i] for i in list(h5_file)])
        
        if preload:
            print('Preloading IXITrainSet')
            for i in range(len(self.samples)):
                self.samples[i] = self.samples[i][:]
        self.torchiosub = torchiosub
        self.lazypatch = lazypatch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        if self.torchiosub:
            return Subject({'img':IXI_H5DSImage(self.samples[item], lazypatch=self.lazypatch)})
        else:
            return torch.from_numpy(self.samples[item][()]).unsqueeze(0)

