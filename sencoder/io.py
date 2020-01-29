import os
import torch
from sencoder.models import *

def save_checkpoint(save_dict, folder, del_prev=False):
    """
    save_dict: dict
        all things to save to file
    folder: str
        path of folder to be saved to
    del_prev: bool
        if true, deletes the model_state_dict and optim_state_dict of the save of the
        previous file (saves space)
    """
    if del_prev:
        prev_path = os.path.join(folder, "epoch_" + str(save_dict['epoch']-1) + '.pt')
        if os.path.exists(prev_path):
            device = torch.device("cpu")
            data = torch.load(prev_path, map_location=device)
            keys = list(data.keys())
            for key in keys:
                if "state_dict" in key:
                    del data[key]
            torch.save(data, prev_path)
        elif save_dict['epoch'] != 0:
            print("Failed to find previous checkpoint", prev_path)
    path = os.path.join(folder, 'epoch_' + str(save_dict['epoch'])) + '.pt'
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)

def load_model(model_path, ret_chkpt=False):
    """
    model_path: str
        can be a .p, .pt, .pth file or a save folder.
    """
    model_path = os.path.expanduser(model_path)
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        chkpts = []
        for f in files:
            if ".p" in f or ".pt" in f or ".pth" in f:
                chkpts.append(f)
        keyfxn = lambda x: int(x.split("_")[-1].split(".")[0])
        chkpts = sorted(chkpts, key=keyfxn)
        model_path = os.path.join(model_path, chkpts[-1])
    assert not os.path.isdir(model_path)
    return load_checkpoint(model_path, ret_chkpt=ret_chkpt)

def load_checkpoint(checkpt_path, ret_chkpt=False):
    """
    Can load a specific model file both architecture and state_dict if the file 
    contains a model_state_dict key, or can just load the architecture.

    checkpt_path: str
        path to checkpoint file
    """
    checkpt_path = os.path.expanduser(checkpt_path)
    data = torch.load(checkpt_path, map_location=torch.device("cpu"))
    model = globals()[data['model_type']](**data['hyps'])
    model.load_state_dict(data['model_state_dict'])
    if ret_chkpt:
        return model, data
    return model

