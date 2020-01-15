import torch
import torch.nn as nn
from torch.nn import LSTMCell
from sencoder.torch_utils import *
import numpy as np
from scipy import signal
import os

DEVICE = torch.device('cuda:0')

def try_kwarg(kwargs, key, default):
    try:
        return kwargs[key]
    except:
        return default

class CustomModule:
    @property
    def is_cuda(self):
        try:
            return next(self.parameters()).is_cuda
        except:
            return False

    def get_device(self):
        try:
            return next(self.parameters()).get_device()
        except:
            return -1

class RSSM(nn.Module, CustomModule):
    def __init__(self, h_size, s_size, emb_size, rnn_type="GRU",
                                            min_sigma=0.0001):
        super().__init__()
        """
        h_size - int
            size of belief vector h
        s_size - int
            size of state vector s
        emb_size - int
            the size of the embedding.

        min_sigma - float
            the minimum value that the state standard deviation can
            take
        """
        if rnn_type == "GRU":
            rnn_type = "GRUCell"
        elif rnn_type == "LSTM":
            rnn_type = "LSTMCell"
        else:
            rnn_type = "DoubleLSTMCell"
        self.h_size = h_size
        self.s_size = s_size
        self.emb_size = emb_size
        self.rnn_type = rnn_type
        self.min_sigma = min_sigma

        self.rnn = globals()[rnn_type](input_size=(s_size+emb_size), hidden_size=h_size) # Dynamics rnn
        self.state_layer = nn.Linear(h_size, 2*s_size) # Creates mu and sigma for state gaussian

    def init_h(self, batch_size):
        if self.rnn_type == "DoubleLSTMCell":
            h = self.rnn.init_h(batch_size)
        else:
            h = torch.zeros(batch_size, self.h_size)
            if self.is_cuda:
                h = h.to(DEVICE)
        if self.rnn_type == "LSTMCell":
            c = torch.zeros_like(h)
            h = (h,c)
        else:
            h = (h,)

        mu = torch.zeros(batch_size, self.s_size)
        if self.is_cuda:
            mu = mu.to(DEVICE)
        sigma = torch.ones_like(mu)
        return h,mu,sigma

    def forward(self, x, h_tup):
        h,mu,sigma = h_tup
        noise = torch.randn(*sigma.shape)
        if sigma.is_cuda:
            noise = noise.to(DEVICE)
        s = sigma*noise+mu
        x = torch.cat([s,x], dim=-1)
        h_new = self.rnn(x, h)
        musigma = self.state_layer(h_new[0])
        mu, sigma = torch.chunk(musigma, 2, dim=-1)
        sigma = F.softplus(sigma) + self.min_sigma
        return h_new, mu, sigma
    
    def extra_repr(self):
        names = ["h_size={}","s_size={}","emb_size={}",
                                        "min_sigma={}"]
        return ", ".join(names).format(self.h_size, self.s_size,
                                    self.emb_size, self.min_sigma)

class Encoder(nn.Module):
    def __init__(self, emb_size, rssm_kwargs, proj_size=500,
                                  proj_layers=2, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.proj_size = proj_size
        self.rssm = RSSM(**rssm_kwargs)
        if try_kwarg(kwargs, 'wnorm', False):
            rnn = self.rssm.rnn
            if self.rssm.rnn_type == "GRUCell":
                rnn = rnn.gru
            names = [name for name,_ in rnn.named_parameters()]
            for name in names:
                if 'bias' not in name:
                    rnn = nn.utils.weight_norm(rnn, name)
        # Projection NN
        modules = []
        in_size = self.rssm.h_size+self.rssm.s_size
        if proj_layers == 1:
            modules.append(nn.Linear(in_size, emb_size))
        else:
            modules.append(nn.Linear(in_size, proj_size))
        for i in range(proj_layers-1):
            modules.append(nn.ReLU())
            if i < proj_layers-2:
                modules.append(nn.Linear(proj_size, proj_size))
            else:
                modules.append(nn.Linear(proj_size, emb_size))
        self.projection = nn.Sequential(*modules)

    def init_h(self, batch_size):
        return self.rssm.init_h(batch_size)

    def forward(self, X, h=None):
        """
        X: torch FloatTensor (B,S,E)
            batch of sequences of embeddings
        h: tuple of FloatTensors ((B,H), (B,C))
            if using LSTM, must include c vector as well as h.
            If None, the state is initialized to zeros
        """
        if h is None:
            h = self.init_h(len(X))
        embs = []
        states = []
        for i in range(X.shape[1]):
            x = X[:,i]
            h = self.rssm(x,h)
            states.append(h)
            noise = torch.randn(h[1].shape)
            if h[1].is_cuda:
                noise = noise.to(DEVICE)
            s = h[1]+h[2]*noise
            x = torch.cat([h[0][0],s], dim=-1)
            emb = self.project(x)
            embs.append(emb)
        return states,embs

    def project(self,x):
        """
        Projects the h vector into embedding space.

        x: torch FloatTensor (B,H)
            most likely want to use h vector as input here
        """
        return self.projection(x)

class Decoder(nn.Module):
    def __init__(self, emb_size, rssm_kwargs, stop_idx,
                            proj_size=500, proj_layers=2,
                            **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.proj_size = proj_size
        self.stop_idx = stop_idx
        self.rssm = RSSM(**rssm_kwargs)
        if try_kwarg(kwargs, 'wnorm', False):
            rnn = self.rssm.rnn
            if self.rssm.rnn_type == "GRUCell":
                rnn = rnn.gru
            names = [name for name,_ in rnn.named_parameters()]
            for name in names:
                if 'bias' not in name:
                    rnn = nn.utils.weight_norm(rnn, name)
        # Projection NN
        modules = []
        in_size = self.rssm.h_size+self.rssm.s_size
        if proj_layers == 1:
            modules.append(nn.Linear(in_size, emb_size))
        else:
            modules.append(nn.Linear(in_size, proj_size))
        for i in range(proj_layers-1):
            modules.append(nn.ReLU())
            if i < proj_layers-2:
                modules.append(nn.Linear(proj_size, proj_size))
            else:
                modules.append(nn.Linear(proj_size, emb_size))
        self.projection = nn.Sequential(*modules)

    def init_h(self, batch_size):
        return self.rssm.init_h(batch_size)

    def forward(self, h, seq_len, embs=None):
        """
        h: tuple of torch FloatTensors [(B,H), (B,S), (B,S)]
            the final state of the encoding RSSM
        seq_len: int
            the number of decoding steps to perfor
        embs: list of torch FloatTensors
            the true sequence of embeddings. Must be of length
            seq_len. If none, rssm uses predicted embeddings.
        """
        if embs is None:
            emb = torch.zeros(len(enc), self.emb_size)
            if next(self.parameters()).is_cuda:
                emb = emb.to(DEVICE)
        preds = [emb]
        states = []
        for i in range(seq_len+1):
            x = preds[i] if embs is None else embs[i]
            h = self.rssm(x,h)
            states.append(h)
            noise = torch.randn(h[1].shape)
            if h[1].is_cuda:
                noise = noise.to(DEVICE)
            s = h[2]*noise + h[1]
            x = torch.cat([h[0][0],s],dim=-1)
            pred = self.project(x)
            preds.append(embs)
        return states,preds[1:]

    def project(self,x):
        """
        Projects the h vector into embedding space.

        x: torch FloatTensor (B,H)
            most likely want to use h vector as input here
        """
        return self.projection(x)

class SeqAutoencoder(nn.Module):
    def __init__(self, emb_size, n_words, stop_idx, **kwargs):
        super().__init__()
        """
        kwargs:
            n_words
        """
        self.emb_size = emb_size
        self.n_words = n_words

        # Embeddings
        std = 2/float(np.sqrt(n_words+emb_size))
        self.embeddings = std*torch.randn(n_words,emb_size)
        self.embeddings = nn.Parameter(self.embeddings)

        # Encoder
        rssm_kwargs = {"h_size": kwargs['h_size'],
                       "s_size": kwargs['s_size'],
                       'emb_size': emb_size,
                       'rnn_type': kwargs['rnn_type']}
        self.encoder = Encoder(emb_size=emb_size,
                             rssm_kwargs=rssm_kwargs, **kwargs)

        # Decoder
        rssm_kwargs = {"h_size": kwargs['h_size'],
                       "s_size": kwargs['s_size'],
                       'emb_size': emb_size,
                       'rnn_type': kwargs['rnn_type']}
        self.decoder = Decoder(emb_size=emb_size,rssm_kwargs=rssm_kwargs,
                                             stop_idx=stop_idx, **kwargs)

        # Classifier
        cl_type = kwargs['classifier_type']
        n_layers = kwargs['classifier_layers']
        self.classifier = globals()[cl_type](emb_size=emb_size,
                                               n_words=n_words,
                                               n_layers=n_layers)

    def enc_requires_grad(self, state):
        for p in self.encoder.parameters():
            p.requires_grad = state

    def class_requires_grad(self, state):
        for p in self.classifier.parameters():
            p.requires_grad = state

    def embed(self, idxs):
        """
        idxs: torch LongTensor (...)
        
        Returns:
            embs: torch FloatTensor (..., E)
        """
        shape = idxs.shape
        idxs = idxs.reshape(-1)
        embs = self.embeddings[idxs]
        embs = embs.reshape(*shape,embs.shape[-1])
        return embs

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def classify(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)

class SimpleClassifier(nn.Module):
    def __init__(self, emb_size, n_words, n_layers=2,**kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.n_words = n_words
        self.classifier = nn.Sequential(nn.Linear(emb_size, n_words//2),
                            nn.ReLU(),nn.Linear(n_words//2,n_words))

    def forward(self, x):
        return self.classifier(x)








