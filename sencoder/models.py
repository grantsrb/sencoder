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

        in_size = (s_size+emb_size)
        self.rnn = globals()[rnn_type](input_size=in_size,
                                        hidden_size=h_size)
        # Creates mu and sigma for state gaussian
        self.state_layer = nn.Linear(h_size, 2*s_size)

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
        s = sigma*torch.randn_like(sigma)+mu
        x = torch.cat([s,x], dim=-1)
        h_new = self.rnn(x, h)
        musigma = self.state_layer(h_new[0])
        mu, sigma = torch.chunk(musigma, 2, dim=-1)
        sigma = F.softplus(sigma) + self.min_sigma
        return h_new, mu, sigma

    def state_fwd(self, x, h_tup, state):
        """
        Same as forward function but includes an extra state
        vector. Improves speed by reducing concatenations.

        x: FloatTesor (B,E)
        h_tup: list or tuple of FloatTensors [rnn_h, (B,S), (B,S)]
            h,mu,sigma
        state: FloatTensor (B,S)
            the encoded state vector
        """
        h,mu,sigma = h_tup
        s = sigma*torch.randn_like(sigma)+mu
        x = torch.cat([s,x,state], dim=-1)
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
    def __init__(self, emb_size, rssm_kwargs, attention=False,
                                                    **kwargs):
        """
        emb_size: int
            the size of the embedding vectors
        
        """
        super().__init__()
        self.emb_size = emb_size
        self.attention = None
        self.rssm = RSSM(**rssm_kwargs)
        if try_kwarg(kwargs, 'wnorm', False):
            rnn = self.rssm.rnn
            if self.rssm.rnn_type == "GRUCell":
                rnn = rnn.gru
            names = [name for name,_ in rnn.named_parameters()]
            for name in names:
                if 'bias' not in name:
                    rnn = nn.utils.weight_norm(rnn, name)
        if attention:
            h_size = rssm_kwargs['h_size']
            s_size = rssm_kwargs['s_size']
            self.attention = nn.Linear(2*(h_size+s_size),1)

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
        hs = []
        mus = torch.zeros(*X.shape[:2],self.rssm.s_size).to(DEVICE)
        sigmas = torch.zeros(*X.shape[:2],self.rssm.s_size).to(DEVICE)
        states = []

        for i in range(X.shape[1]):
            x = X[:,i]
            h = self.rssm(x,h)
            s = h[2]*torch.randn_like(h[2])+h[1]
            hs.append(h[0])
            mus[:,i] = h[1]
            sigmas[:,i] = h[2]

            if self.attention is not None:
                scores = []
                context = torch.cat([h[0][0],s], dim=-1)
                for j in range(len(hs)):
                    h, mu, sigma = hs[j][0][0], hs[j][1], hs[j][2]
                    s = sigma*torch.randn_like(sigma)+mu
                    x = torch.cat([context,h,s],dim=-1)
                    score = self.attention(x)
                    scores.append(score)
                temp = torch.cat(scores, dim=-1)
                alphas = F.softmax(temp, dim=-1)
                ss = torch.cat([mus[:,:i+1],sigmas[:,:i+1]], dim=-1)
                state = torch.einsum("bsh,bs->bh", ss, alphas)
            else:
                state = torch.cat([h[1],h[2]], dim=-1)
            states.append(state)
        return hs,mus,sigmas,states

class Decoder(nn.Module):
    def __init__(self, emb_size, rssm_kwargs, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.rssm = RSSM(**rssm_kwargs)
        if try_kwarg(kwargs, 'wnorm', False):
            rnn = self.rssm.rnn
            if self.rssm.rnn_type == "GRUCell":
                rnn = rnn.gru
            names = [name for name,_ in rnn.named_parameters()]
            for name in names:
                if 'bias' not in name:
                    rnn = nn.utils.weight_norm(rnn, name)

    def init_h(self, batch_size):
        return self.rssm.init_h(batch_size)

    def forward(self, state, h, seq_len, embs=None, classifier=None,
                                                   embeddings=None):
        """
        h: tuple of torch FloatTensors [(B,H), (B,S), (B,S)]
            the final state of the encoding RSSM. So it should
            actually be (h, mu, sigma). Forgive the naming
            abuse
        seq_len: int
            the number of decoding steps to perfor
        embs: list of torch FloatTensors [S length (B,E)]
            the true sequence of embeddings. Must be of length
            seq_len. If none, rssm uses predicted embeddings.
            Must argue classifier and embeddings if embs is None.
        classifier: nn.Module
            if embs is None, then there must be a classifier.
            The classifier makes word predictions from the
            h and s vectors.
        embeddings: nn Parameter (N, E)
            The real embeddings to be used as a representation
            of strings. Must be included if embs is None.
        """
        x = torch.zeros(len(h[0][0]), self.emb_size)
        if next(self.parameters()).is_cuda:
            x = x.to(DEVICE)

        hs = []
        mus = []
        sigmas = []
        s_size = state.shape[1]//2
        for i in range(seq_len):
            s_mu, s_sig = torch.chunk(state,2,dim=-1)
            s = s_sig*torch.randn_like(s_sig)+ s_mu
            h,mu,sigma = self.rssm.state_fwd(x,h,s)
            hs.append(h)
            mus.append(mu)
            sigmas.append(sigma)
            h = (h,mu,sigma)
            if embs is None:
                pred = classifier(mu,sigma)
                idxs = torch.argmax(pred, dim=-1).long()
                x = embeddings[idxs]
            else:
                x = embs[i]
        return hs,mus,sigmas

class WordEncoder(nn.Module):
    def __init__(self, emb_size, s_size, min_sigma=0.0001,
                                                **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.s_size = s_size

        self.encoder = nn.Linear(emb_size, s_size*2)

    def forward(self, x):
        """
        x: torch FloatTensor (B,E)
        """
        musigma = self.encoder(x)
        mu,sigma = torch.chunk(musigma,2,dim=-1)
        sigma = F.softplus(sigma)
        return mu, sigma

class SeqAutoencoder(nn.Module):
    """
    A sequence based autoencoder.
    """
    def __init__(self, emb_size, n_words, h_size=300, s_size=300,
                                      attention=False, **kwargs):
        super().__init__()
        """
        emb_size: int
            size of the embedding dimension
        n_words: int
            number of unique embeddings to create
        h_size: int
            size of the deterministic hidden state in the recurrent
            network
        s_size: int
            size of the stochastic hidden state in the recurrent
            network
        attention: bool
            if true, uses an attention mechanism to create the final
            encoding. If false, uses the final output of the encoder
            as the encoding.
        """
        self.emb_size = emb_size
        self.h_size = h_size
        self.s_size = s_size
        self.n_words = n_words
        self.attention = attention

        # Embeddings
        std = 2/float(np.sqrt(n_words+emb_size))
        self.embeddings = std*torch.randn(n_words,emb_size)
        self.embeddings = nn.Parameter(self.embeddings)

        # Word Encoder
        self.word_encoder = WordEncoder(emb_size, s_size)

        # Encoder
        rssm_kwargs = {"h_size": h_size,
                       "s_size": s_size,
                       'emb_size': emb_size,
                       'rnn_type': kwargs['rnn_type']}
        self.encoder = Encoder(emb_size=emb_size,
                                    attention=attention,
                                    rssm_kwargs=rssm_kwargs,
                                    **kwargs)

        # Decoder
        rssm_kwargs = {"h_size": h_size,
                       "s_size": s_size,
                       'emb_size': emb_size+s_size,
                       'rnn_type': kwargs['rnn_type']}
        self.decoder = Decoder(emb_size=emb_size,
                                    rssm_kwargs=rssm_kwargs,
                                    **kwargs)

        # Classifier
        cl_type = kwargs['classifier_type']
        n_layers = kwargs['classifier_layers']
        self.classifier = globals()[cl_type](s_size=s_size,
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

    def encode_word(self, *args, **kwargs):
        return self.word_encoder(*args, **kwargs)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def classify(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)

class StateClassifier(nn.Module):
    def __init__(self, s_size, n_words, n_layers=2,**kwargs):
        super().__init__()
        self.s_size = s_size
        self.n_words = n_words
        modules = []
        modules.append(nn.Linear(s_size, n_words//2))
        modules.append(nn.ReLU())
        for i in range(1,n_layers-1):
            modules.append(nn.Linear(n_words//2, n_words//2))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(n_words//2, n_words))
        self.classifier = nn.Sequential(*modules)

    def forward(self, s):
        """
        s: torch FloatTensor (B,N)
            the state as encoded by the encoder (most likely using
            self attention)
        """
        return self.classifier(s)

class MuSigClassifier(nn.Module):
    def __init__(self, s_size, n_words, n_layers=2,**kwargs):
        super().__init__()
        self.s_size = s_size
        self.n_words = n_words
        modules.append(nn.Linear(s_size, n_words//2))
        modules.append(nn.ReLU())
        for i in range(1,n_layers-1):
            modules.append(nn.Linear(n_words//2, n_words//2))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(n_words//2, n_words))
        self.classifier = nn.Sequential(*modules)

    def forward(self, mu, sigma):
        """
        mu: torch FloatTensor (B,N)
            the mean of a gaussian as encoded by the encoder or
            decoded by the decoder.
        sigma: torch FloatTensor (B,N)
            the std of a gaussian as encoded by the encoder or
            decoded by the decoder.
        """
        s = sigma*torch.randn_like(sigma)+mu
        return self.classifier(s)








