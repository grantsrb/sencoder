import torch
import sencoder as sen
import numpy as np
import os
import tokenizer as tok
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


sample_strs = {"greetings":["hello kind sir, how are you doing today?",
                            "hi, what is going on?",
                            "good morning dear, It is a pleasure to see you",
                            "how is it going?"],
                "commands":["do not go into the forest.",
                            "you must cut down this tree.",
                            "build a house from the wood.",
                            "go into that cave."
                            ],
                "ramblings":["it is god's will, to be killed or to be lived.",
                            "the subject of good an evil cannot be contained.",
                            "the herd will not see the truths that stare them in the face.",
                            "death will come to those who wait.",
                            "life is tedious and brief."
                            ]
                }

tokens = {k:[tok.tokenize(s) for s in v] for k,v in sample_strs.items()}

for k,v in tokens.items():
    print(k)
    print(v)
    print()

hyps = {"dataset":"Nietzsche","seq_len":30, "shuffle":True,
                           "batch_size":100, "n_workers":1}
train_distr, _ = sen.training.get_data_distrs(hyps)
word2idx = train_distr.dataset.word2idx
idx2word = train_distr.dataset.idx2word

wordidxs = {k:[[word2idx[w] for w in t] for t in v]\
                            for k,v in tokens.items()}
for k,v in wordidxs.items():
    print(k)
    for idxs in v:
        print(idxs)
    print()

prepath = "../training_scripts/"
s = "alldecsteps/alldecsteps_0_lr0.005"
s = os.path.join(prepath,s)
model, chkpt = sen.utils.load_model(s, ret_chkpt=True)
model.to(DEVICE)
model.eval()

states = {}
for k,v in wordidxs.items():
    states[k] = []
    for idxs in v:
        idxs = torch.LongTensor(idxs).to(DEVICE)
        print("idxs:", idxs.shape)
        embs = model.embed(idxs)
        embs = embs.reshape(1,len(idxs),-1)
        print("Embs:", embs.shape)
        enc_hs, enc_mus, enc_sigmas = model.encode(embs)
        state = Normal(enc_mus[-1], enc_sigmas[-1])
        states.append(state)











