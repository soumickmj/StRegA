import h5py as h5
import numpy as np

import torch
from torch.utils.data import Dataset
from torchio.data.subject import Subject
from datasets.ixi_torchiowrap import IXI_H5DSImage

class IXITrainSet(Dataset):
    def __init__(self, indices=None,data_path='Ixi_with_skull.h5', torchiosub=True, lazypatch=True, preload=False):
        self.h5 = h5.File(data_path, 'r', swmr=True)
        self.samples = []
        if indices:
            self.samples = [self.h5[str(i).zfill(5)]for i in indices]
            # self.samples2 = [self.h5[region][str(i).zfill(5)][:] for i in indices]
        else:
            self.samples = [self.h5[i] for i in list(self.h5[region])]
        if preload:
            print('Preloading MoodTrainSet')
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

