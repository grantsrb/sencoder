import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import split_token, tokenize
from keras.utils.data_utils import get_file
import os
import copy

class Nietzsche(Dataset):
    def __init__(self, seq_len=10):
        file_name = "nietzsche.txt"
        url = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
        tup = self.get_data(file_name=file_name, url=url,
                                         seq_len=seq_len)
        X,Y,word2idx,idx2word = tup
        self.X = X # (N, SeqLen)
        self.Y = Y # (N, SeqLen)
        self.word2idx = word2idx
        self.idx2word = idx2word

    def get_data(self, file_name, url, seq_len, **kwargs):
        # Get and prepare data
        data_path = get_file(file_name, origin=url)
        data = open(data_path, 'r')
    
        text = tokenize(data.read())
        words = set(text)
        print("Num unique words:", len(words))
    
        word2idx = {w:i for i,w in enumerate(words)}
        idx2word = {i:w for i,w in enumerate(words)}
    
        X = [[word2idx[text[i+j]] for j in range(seq_len)]\
                            for i in range(len(text)-seq_len-1)]
        Y = [[word2idx[text[i+j]] for j in range(seq_len)]\
                            for i in range(1,len(text)-seq_len)]
        X = torch.LongTensor(X)
        Y = torch.LongTensor(Y)

        assert len(X) == len(Y)
        return X, Y, word2idx, idx2word
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class EmptyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_data_split(train_p=.9, dataset="Nietzsche", seq_len=10):
    """
    Returns two torch Datasets, one validation and one training.

    train_p: float between 0 and 1
        the portion of data used for training
    dataset: str
        the name of the desired dataset
    seq_len: int
        the length of word sequences
    """

    dataset_class = globals()[dataset]
    dataset = dataset_class(seq_len=seq_len)
    perm = torch.randperm(len(dataset)).long()
    split_idx = int(len(perm)*train_p)
    train_idxs = perm[:split_idx]
    val_idxs = perm[split_idx:]

    val_dataset = EmptyDataset(X=dataset.X, Y=dataset.Y)
    val_dataset.X = val_dataset.X[val_idxs]
    val_dataset.Y = val_dataset.Y[val_idxs]

    dataset.X = dataset.X[train_idxs]
    dataset.Y = dataset.Y[train_idxs]
    return dataset, val_dataset








