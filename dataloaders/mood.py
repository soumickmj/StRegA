import h5py as h5
import numpy as np

import torch
from torch.utils.data import Dataset
from torchio.data.subject import Subject

from .torchiowrap import H5DSImage

class MoodTrainSet(Dataset):
    def __init__(self, indices=None, region='brain', data_path='MOOD_train.h5', torchiosub=True, lazypatch=True, preload=False):
        # Support both single path and list of paths
        if isinstance(data_path, str):
            data_path = [data_path]
        
        self.h5_files = [h5.File(path, 'r', swmr=True) for path in data_path]
        self.samples = []
        
        for h5_file in self.h5_files:
            if indices:
                self.samples.extend([h5_file[region][str(i).zfill(5)] for i in indices])
            else:
                self.samples.extend([h5_file[region][i] for i in list(h5_file[region])])
        
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
            return Subject({'img':H5DSImage(self.samples[item], lazypatch=self.lazypatch)})
        else:
            return torch.from_numpy(self.samples[item][()]).unsqueeze(0)

class MoodValSet(Dataset):
    def __init__(self, load_abnormal=True, load_normal=True, loadASTrain=False, data_path='MOOD_val.h5', torchiosub=True, lazypatch=True, preload=False):
        # Support both single path and list of paths
        if isinstance(data_path, str):
            data_path = [data_path]
        
        self.h5_files = [h5.File(path, 'r', swmr=True) for path in data_path]
        self.samples = []
        
        for h5_file in self.h5_files:
            if load_abnormal:
                self.samples += [(h5_file['abnormal'][i], h5_file['abnormal_mask'][i]) for i in list(h5_file['abnormal'])]
            if load_normal:
                self.samples += [h5_file['normal'][i] for i in list(h5_file['normal'])]
        
        if preload:
            print('Preloading MoodValSet')
            for i in range(len(self.samples)):
                if len(self.samples[i]) == 2:
                    self.samples[i] = (self.samples[i][0][:], self.samples[i][1][:])
                else:
                    self.samples[i] = self.samples[i][:]
        self.loadASTrain = loadASTrain
        self.torchiosub = torchiosub
        self.lazypatch = lazypatch

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        if self.loadASTrain:
            if self.torchiosub:
                return Subject({'img':H5DSImage(self.samples[item][0], lazypatch=self.lazypatch)})
            else:
                return torch.from_numpy(self.samples[item][0][()]).unsqueeze(0)
        else:
            if self.torchiosub:
                if len(self.samples[item]) == 2:
                    return Subject({'img':H5DSImage(self.samples[item][0], lazypatch=self.lazypatch),
                                    'gt':H5DSImage(self.samples[item][1], lazypatch=self.lazypatch)})
                else:
                    return Subject({'img':H5DSImage(self.samples[item], lazypatch=self.lazypatch),
                                    'gt':H5DSImage(self.samples[item], lazypatch=self.lazypatch)}) #this is dirty. TODO

            else:
                return (torch.from_numpy(self.samples[item][0][()]).unsqueeze(0), torch.from_numpy(self.samples[item][1][()]).unsqueeze(0))