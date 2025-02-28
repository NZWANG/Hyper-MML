import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy
class IEMOCAPDataset(Dataset):

    def __init__(self, train=True):
        # 从pickle文件中加载IEMOCAP数据集
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

        _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
        _, _, _, _ = pickle.load(open('./IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''
        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array([[1,0] if x=='M' else [0,1] for x in\
                                  self.videoSpeakers[vid]])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        """自定义批处理函数"""
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]


class EAVDataset(Dataset):
    def __init__(self, train=True, subject=1):
        '''
            label index mapping = {'Neutral':0, 'Sadness':1, 'Anger':2, 'Happiness':3, 'Calmness':4}
        '''
        # 从pickle文件中加载EAV数据集
        self.subject = subject
        self.tr_aud, self.te_aud = pickle.load(open(f'/root/autodl-tmp/EAV/Features/subject{subject:02d}/subject_{subject:02d}_aud.pkl', 'rb'))
        self.tr_vis, self.te_vis = pickle.load(open(f'/root/autodl-tmp/EAV/Features/subject{subject:02d}/subject_{subject:02d}_vis.pkl', 'rb'))
        self.tr_eeg, self.te_eeg = pickle.load(
            open(f'/root/autodl-tmp/EAV/Features/subject{subject:02d}/subject_{subject:02d}_eeg.pkl', 'rb'))
        self.tr_lab, self.te_lab = pickle.load(
            open(f'/root/autodl-tmp/EAV/Features/subject{subject:02d}/subject_{subject:02d}_lab.pkl', 'rb'))
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
        """自定义批处理函数"""
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open(path, 'rb'))

        _, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            _, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open("./MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
               torch.FloatTensor(numpy.array(self.roberta2[vid])),\
               torch.FloatTensor(numpy.array(self.roberta3[vid])),\
               torch.FloatTensor(numpy.array(self.roberta4[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]
