import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy

class EAVDataset(Dataset):
    def __init__(self, train=True, subject=1):
        '''
            label index mapping = {'Neutral':0, 'Sadness':1, 'Anger':2, 'Happiness':3, 'Calmness':4}
        '''
        self.subject = subject
        self.tr_aud, self.te_aud = pickle.load(open(f'/root/autodl-tmp/dataset/EAV/features/Audio_openSMILE/subject_{subject:02d}_aud.pkl', 'rb'))
        self.tr_vis, self.te_vis = pickle.load(open(f'/root/autodl-tmp/dataset/EAV/features/Vision_manet/subject_{subject:02d}_vis.pkl', 'rb'))
        self.tr_eeg, self.te_eeg = pickle.load(
            open(f'/root/autodl-tmp/dataset/EAV/features/EEG_NESTA/subject_{subject:02d}_eeg.pkl', 'rb'))
        self.tr_lab, self.te_lab = pickle.load(
            open(f'/root/autodl-tmp/dataset/EAV/features/lab/subject_{subject:02d}_lab.pkl', 'rb'))
        self.train = train
        if self.train:
            self.len = len(self.tr_lab)
        else:
            self.len = len(self.te_lab)

    def __getitem__(self, index):
        if self.train:
            return torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_vis[index])), \
                torch.FloatTensor(numpy.array(self.tr_aud[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.tr_lab[index]])),\
                index
        else:
            return torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_vis[index])), \
                torch.FloatTensor(numpy.array(self.te_aud[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.te_lab[index]])), \
                index

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class AFFECDataset(Dataset):
    def __init__(self, train=True, subject='1'):

        self.subject = subject
        if subject == 1:
            user = 'acl'
        self.tr_eye, self.te_eye = pickle.load(open(f'/root/autodl-tmp/dataset/AFFEC/feature/sub-{user}/sub-{user}_eye.pkl', 'rb'))
        self.tr_gsr, self.te_gsr = pickle.load(open(f'/root/autodl-tmp/dataset/AFFEC/feature/sub-{user}/sub-{user}_gsr.pkl', 'rb'))
        self.tr_eeg, self.te_eeg = pickle.load(
            open(f'/root/autodl-tmp/dataset/AFFEC/feature/sub-{user}/sub-{user}_eeg.pkl', 'rb'))
        _, _, self.tr_lab, _, _, _, self.te_lab, _ = pickle.load(
            open(f'/root/autodl-tmp/dataset/AFFEC/feature/sub-{user}/sub-{user}_lab.pkl', 'rb'))
        self.train = train
        if self.train:
            self.len = len(self.tr_lab)
        else:
            self.len = len(self.te_lab)

    def __getitem__(self, index):
        if self.train:
            return torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eye[index])), \
                torch.FloatTensor(numpy.array(self.tr_gsr[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.tr_lab[index]])),\
                index
        else:
            return torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eye[index])), \
                torch.FloatTensor(numpy.array(self.te_gsr[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.te_lab[index]])), \
                index

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]
