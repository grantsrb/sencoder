import os
import sys
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim.lr_scheduler import *
import sencoder.datas as datas
import sencoder.io as io
from sencoder.models import *
import time
import math
from queue import Queue
import gc
import resource
import json

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def record_session(model, hyps):
    """
    model: torch nn.Module
        the model to be trained
    hyps: dict
        dict of relevant hyperparameters
    """
    if not os.path.exists(hyps['save_folder']):
        os.mkdir(hyps['save_folder'])
    with open(os.path.join(hyps['save_folder'],"hyperparams.txt"),'w') as f:
        f.write(str(model)+'\n')
        for k in sorted(hyps.keys()):
            f.write(str(k) + ": " + str(hyps[k]) + "\n")
    with open(os.path.join(hyps['save_folder'],"hyperparams.json"),'w') as f:
        temp_hyps = {k:v for k,v in hyps.items()}
        json.dump(temp_hyps, f)

def get_model(hyps):
    """
    Gets the model

    hyps: dict
        dict of relevant hyperparameters
    """
    model = globals()[hyps['model_type']](**hyps)
    model = model.to(DEVICE)
    return model

def get_optim_objs(hyps, model):
    """
    hyps: dict
        dict of relevant hyperparameters
    model: torch nn.Module
        the model to be trained
    """
    enc_lossfxn = globals()[hyps['enc_lossfxn']]()
    dec_lossfxn = globals()[hyps['dec_lossfxn']]()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=hyps['lr'],
                                weight_decay=hyps['l2'])
    if 'scheduler' in hyps and "n_epochs" in hyps and\
                       hyps['scheduler'] == "CosineAnnealingLR":
        scheduler = globals()[hyps['scheduler']](optimizer,
                                    T_max=hyps['n_epochs'],
                                    eta_min=5e-5)
    else:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5,
                                                 patience=8)
    return optimizer, scheduler, enc_lossfxn, dec_lossfxn

def print_train_update(loss, acc, enc_loss, dec_loss, n_loops, i):
    s = "Loss:{:.4e} | Acc:{:.4e} | enc:{:.4e} | dec:{:.4e} | {}/{}".format(loss,
                                            acc, enc_loss, dec_loss, i, n_loops)
    print(s, end="       \r")

def get_data_distrs(hyps):
    dataset = hyps['dataset']
    seq_len = hyps['seq_len']
    shuffle = hyps['shuffle']
    batch_size = hyps['batch_size']

    n_workers = 3 if 'n_workers' not in hyps else\
                                    hyps['n_workers']
    train_data, val_data = datas.get_data_split(train_p=0.9,
                                            dataset=dataset,
                                            seq_len=seq_len)
    print("Train Len:", len(train_data))
    print("Val Len:", len(val_data))

    train_distr = torch.utils.data.DataLoader(train_data,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=n_workers)
    val_distr = torch.utils.data.DataLoader(val_data,
                                 batch_size=batch_size,
                                 shuffle=False, num_workers=0)
    return train_distr, val_distr

def train(hyps, verbose=False):
    """
    hyps: dict
        all the hyperparameters set by the user
    verbose: bool
        if true, will print status updates during training
    """
    # Initialize miscellaneous parameters 
    torch.cuda.empty_cache()
    batch_size = hyps['batch_size']

    # Get Data Distributers
    train_distr, val_distr = get_data_distrs(hyps)

    hyps["n_words"] = len(train_distr.dataset.word2idx)
    model = get_model(hyps)

    record_session(model, hyps)

    # Make optimization objects (lossfxn, optimizer, scheduler)
    tup = get_optim_objs(hyps, model)
    optimizer, scheduler, enc_lossfxn, dec_lossfxn = tup

    # Training
    epoch = -1
    while hyps['n_epochs'] is None or epoch < hyps['n_epochs']:
        epoch += 1
        print("Epoch", epoch, " -- ", hyps['save_folder'])
        n_loops = len(train_distr)
        model.train()
        model.enc_requires_grad(True)
        if hyps['train_in_eval']:
            model.eval()

        epoch_loss = 0
        epoch_acc = 0
        alpha = hyps['dec_alpha']
        starttime = time.time()
        stats_string = 'Epoch {} -- {}\n'.format(epoch,
                                   hyps['save_folder'])
        # Batch Loop
        optimizer.zero_grad()
        seq_len = hyps['seq_len']
        dec_losses = [0 for i in range(seq_len)]
        #global_h = model.encoder.init_h(batch_size)
        for i,(X,y) in enumerate(train_distr):
            """
            X: torch FloatTensor (B,S)
                batch of idx sequences. S is seq_len
            y: torch FloatTensor (B,S)
                batch of idx sequences shfited by 1.
            """
            if len(y) <= 1:
                i -= 1
                break
            iterstart = time.time()
            y = y.long().to(DEVICE).reshape(-1)
            X = X.long()
            embs = model.embed(X) # (B,S,E)

            # Make next word predictions with encoder
            tup = model.encode(embs,h)
            enc_hs, enc_mus, enc_sigmas, enc_states = tup
            global_h = enc_hs[0]
            s_size = hyps['s_size']
            states = torch.stack(enc_states,dim=1)
            states = states.reshape(-1,states.shape[-1])
            enc_preds = model.classify(states)
            enc_loss = enc_lossfxn(enc_preds, y) # scalar

            # Decoder decodes for every dec_step_size steps. We make
            # sure to include the maximum number of encoded steps.
            # This is why the for loop looks complicated
            dec_loss = torch.zeros(1).to(DEVICE)
            s = 'dec_step_size'
            rng = list(range(0,len(enc_hs)-1,hyps[s]))+[len(enc_hs)-1]
            for j in rng:
                state = enc_states[j]
                h = (enc_hs[j],enc_mus[:,j],enc_sigmas[:,j])
                hs,mus,sigmas = model.decode(state, h, seq_len=j+1,
                                        classifier=model.classifier,
                                        embeddings=model.embeddings)
                mus = torch.stack(mus, dim=1)
                mus = mus.reshape(-1,s_size)
                sigmas = torch.stack(sigmas, dim=1)
                sigmas = sigmas.reshape(-1,s_size)
                states = sigmas*torch.randn_like(sigmas)+mus
                dec_preds = model.classify(states)
                targs = torch.flip(X[:,:j+1],dims=(1,))
                targs = targs.to(DEVICE).reshape(-1)
                loss = dec_lossfxn(dec_preds, targs)/(j+1)
                dec_losses[j] += loss.item()
                dec_loss += loss
                # TODO: include probability distance loss (probably
                # KL Divergence) between decoder states and encoder
                # states.
            loss = alpha*dec_loss + (1-alpha)*enc_loss
            #loss = enc_loss
            loss = loss/hyps['optim_batches']
            loss.backward()
            if hyps['optim_batches'] == 1 or\
                    (i > 0 and i%hyps['optim_batches'] == 0):
                optimizer.step()
                optimizer.zero_grad()

            argmaxes=torch.argmax(enc_preds,dim=-1).long()
            acc = (argmaxes.squeeze()==y).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if verbose:
                looptime = time.time()-iterstart
                print_train_update(loss.item(), acc.item(),
                         enc_loss.item(), dec_loss.item(),
                         n_loops, i)
            if math.isnan(epoch_loss) or math.isinf(epoch_loss)\
                                    or hyps['exp_name']=="test":
                break
        # Clean Up Train Loop
        n_loops = i+1 # Just in case miscalculated
        avg_loss = epoch_loss/n_loops
        avg_acc = epoch_acc/n_loops
        dec_avgs = ["{:.3e}".format(d/n_loops) for d in dec_losses]
        looptime = time.time()-starttime
        s = 'Avg Loss: {} | Avg Acc: {} | Time: {}\n'.format(
                                            avg_loss, avg_acc,
                                            looptime)
        s += "DecLosses:" + "|".join(dec_avgs) + "\n"
        stats_string += s

        # Validation
        model.eval()
        starttime = time.time()
        n_loops = len(val_distr)
        val_loss = 0
        val_acc  = 0
        dec_val_acc = 0
        if verbose:
            print()
            print("Validating")

        # Val Loop
        with torch.no_grad():
            for i,(X,y) in enumerate(val_distr):
                if len(y) <= 1:
                    i -= 1
                    break
                iterstart = time.time()
                y_shape = y.shape
                y = y.long().to(DEVICE).reshape(-1)
                X = X.long()
                embs = model.embed(X)

                tup = model.encode(embs)
                enc_hs, enc_mus, enc_sigmas, enc_states = tup
                s_size = hyps['s_size']
                mus = enc_mus.reshape(-1,s_size)
                sigmas = enc_sigmas.reshape(-1,s_size)
                enc_preds = model.classify(mus, sigmas)
                enc_loss = enc_lossfxn(enc_preds, y) # scalar

                state = enc_states[-1]
                h = (enc_hs[-1],enc_mus[:,-1],enc_sigmas[:,-1])
                hs,mus,sigmas = model.decode(state,h,seq_len=seq_len,
                                        classifier=model.classifier,
                                        embeddings=model.embeddings)
                mus = torch.stack(mus, dim=1)
                mus = mus.reshape(-1,s_size)
                sigmas = torch.stack(sigmas, dim=1)
                sigmas = sigmas.reshape(-1,s_size)
                dec_preds = model.classify(mus, sigmas)
                targs = torch.flip(X[:,:],dims=(1,))
                targs = targs.to(DEVICE).reshape(-1)
                dec_loss = dec_lossfxn(dec_preds, targs)

                loss = alpha*dec_loss + (1-alpha)*enc_loss
                loss = loss/hyps['optim_batches']

                argmaxes=torch.argmax(enc_preds,dim=-1).long()
                acc = (argmaxes.squeeze()==y).float().mean()
                decmaxes=torch.argmax(dec_preds,dim=-1).long()
                dec_acc = (decmaxes.squeeze()==targs).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()
                dec_val_acc += dec_acc.item()
                if verbose:
                    print_train_update(loss.item(), acc.item(),
                             enc_loss.item(), dec_loss.item(),
                             n_loops, i)
                if math.isnan(loss) or math.isinf(loss) or\
                                    hyps['exp_name']=="test":
                    break
        print()
        n_loops = i+1 # Just in case miscalculated

        # Validation Evaluation
        val_loss = val_loss/n_loops
        val_acc = val_acc/n_loops
        dec_val_acc = dec_val_acc/n_loops
        looptime = time.time()-starttime
        s = 'Val Loss: {} | Val Acc: {} | Dec Acc: {} | Time: {}\n'
        stats_string += s.format(val_loss, val_acc, dec_val_acc,
                                                       looptime)

        idx2word = train_distr.dataset.idx2word
        reals = y.reshape(y_shape)[0]
        real_words = [idx2word[arg.item()] for arg in reals]
        real_str = " ".join(real_words)
        stats_string += "Real: {}\n".format(real_str)

        encs = argmaxes.reshape(y_shape)[0]
        enc_words = [idx2word[arg.item()] for arg in encs]
        enc_str = " ".join(enc_words)
        stats_string += "Enc: {}\n".format(enc_str)

        decmaxes = decmaxes.reshape(y_shape)[0]
        words = [idx2word[arg.item()] for arg in decmaxes]
        dec_words = reversed(words)
        dec_str = " ".join(dec_words)
        stats_string += "Dec: {}\n".format(dec_str)

        if 'scheduler' in hyps and\
                       hyps['scheduler'] == "CosineAnnealingLR":
            scheduler.step()
        elif 'scheduler' in hyps and\
                       hyps['scheduler'] == "ReduceLROnPlateau":
            scheduler.step(val_loss)

        # Save Model Snapshot
        optimizer.zero_grad()
        save_dict = {
            "model_type": hyps['model_type'],
            "model_state_dict":model.state_dict(),
            "optim_state_dict":optimizer.state_dict(),
            "loss": avg_loss,
            "acc": avg_acc,
            "epoch":epoch,
            "val_loss":val_loss,
            "val_acc":val_acc,
            "hyps":hyps,
            "word2idx":train_distr.dataset.word2idx,
            "idx2word":train_distr.dataset.idx2word,
        }
        for k in hyps.keys():
            if k not in save_dict:
                save_dict[k] = hyps[k]
        io.save_checkpoint(save_dict, hyps['save_folder'], del_prev=True)

        # Print Epoch Stats
        gc.collect()
        max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stats_string += "Memory Used: {:.2f} mb".format(max_mem_used / 1024)+"\n"
        print(stats_string)
        with open(os.path.join(hyps['save_folder'],"training_log.txt"),'a') as f:
            f.write(str(stats_string)+'\n')
        # If loss is nan, training is futile
        if math.isnan(avg_loss) or math.isinf(avg_loss) or hyps['exp_name']=="test":
            break

    # Final save
    results = {
                "save_folder":hyps['save_folder'], 
                "Loss":avg_loss, 
                "Acc":avg_acc,
                "ValAcc":val_acc, 
                "ValLoss":val_loss 
                }
    with open(hyps['save_folder'] + "/hyperparams.txt",'a') as f:
        s = " ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
        s = "\n" + s + '\n'
        f.write(s)
    return results

def fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0):
    """
    Recursive function to load each of the hyperparameter combinations 
    onto a queue.

    hyps - dict of hyperparameters created by a HyperParameters object
        type: dict
        keys: name of hyperparameter
        values: value of hyperparameter
    hyp_ranges - dict of ranges for hyperparameters to take over the search
        type: dict
        keys: name of hyperparameters to be searched over
        values: list of values to search over for that hyperparameter
    keys - keys of the hyperparameters to be searched over. Used to
            specify order of keys to search
    train - method that handles training of model. Should return a dict of results.
    hyper_q - Queue to hold all parameter sets
    idx - the index of the current key to be searched over
    """
    # Base call, runs the training and saves the result
    if idx >= len(keys):
        if 'exp_num' not in hyps:
            if 'starting_exp_num' not in hyps: hyps['starting_exp_num'] = 0
            hyps['exp_num'] = hyps['starting_exp_num']
        hyps['save_folder'] = hyps['exp_name'] + "/" + hyps['exp_name'] +"_"+ str(hyps['exp_num']) 
        for k in keys:
            hyps['save_folder'] += "_" + str(k)+str(hyps[k])

        # Load q
        hyper_q.put({k:v for k,v in hyps.items()})
        hyps['exp_num'] += 1

    # Non-base call. Sets a hyperparameter to a new search value and passes down the dict.
    else:
        key = keys[idx]
        for param in hyp_ranges[key]:
            hyps[key] = param
            hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx+1)
    return hyper_q

def hyper_search(hyps, hyp_ranges, keys, device, early_stopping=10,
                                               stop_tolerance=.01):
    starttime = time.time()
    # Make results file
    if not os.path.exists(hyps['exp_name']):
        os.mkdir(hyps['exp_name'])
    results_file = hyps['exp_name']+"/results.txt"
    with open(results_file,'a') as f:
        f.write("Hyperparameters:\n")
        for k in hyps.keys():
            if k not in hyp_ranges:
                f.write(str(k) + ": " + str(hyps[k]) + '\n')
        f.write("\nHyperranges:\n")
        for k in hyp_ranges.keys():
            f.write(str(k) + ": [" + ",".join([str(v) for v in hyp_ranges[k]])+']\n')
        f.write('\n')
    
    hyper_q = Queue()
    
    hyper_q = fill_hyper_q(hyps, hyp_ranges, keys, hyper_q, idx=0)
    total_searches = hyper_q.qsize()
    print("n_searches:", total_searches)

    result_count = 0
    print("Starting Hyperloop")
    while not hyper_q.empty():
        print("Searches left:", hyper_q.qsize(),"-- Running Time:", time.time()-starttime)
        print()
        hyps = hyper_q.get()
        results = train(hyps, verbose=True)
        with open(results_file,'a') as f:
            results = " -- ".join([str(k)+":"+str(results[k]) for k in sorted(results.keys())])
            f.write("\n"+results+"\n")

