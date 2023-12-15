from itertools import chain
import collections
from pathlib import Path
import shutil

import numpy as np

import torch
from torch.utils.data.dataloader import default_collate


# processing applied to hyperparameters
def process_hparams(args):
    args.data = Path(args.data)
    args.log_dir = Path(args.log_dir)
    args.checkpoint_dir = Path(args.checkpoint_dir)

    args.hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    return args

# 
def convert_token_to_label(tokens, token_to_label):
    idxs = []
    for i in range(len(tokens)):
        if tokens[i] not in token_to_label:
            token_to_label[tokens[i]] = len(token_to_label)

        idxs.append(token_to_label[tokens[i]])

    return idxs

# pad list of strings to same length, pad obs_idxs with the last state too
def pad_list_of_strings(list_of_strings, obs_idxs, pad_token='<pad>'):
    max_len = max([len(s) for s in list_of_strings])
    
    for l in list_of_strings:
        l += [f'{pad_token}'] * (max_len - len(l))
    
    return list_of_strings

# # custom collate designed for new dataset without obs idxs (classifier)
def custom_collate_clf(batch):
    elem = batch[0]

    if isinstance(elem, collections.abc.Sequence): # custom collate for subgoals (assuming it is index 1 in the batch)
        # check to make sure that the elements in batch have consistent size
        # it = iter(batch)
        # elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))

        return [default_collate(transposed[0]), transposed[1], default_collate(transposed[2]), default_collate(transposed[3])] # task, subgoals, obs, label
        
    else:  # Fall back to `default_collate`
        return default_collate(batch)

# custom collate designed for new dataset (scorer)
def custom_collate_scorer(batch):
    elem = batch[0]

    if isinstance(elem, collections.abc.Sequence): # custom collate for subgoals (assuming it is index 1 in the batch)
        # check to make sure that the elements in batch have consistent size
        # it = iter(batch)
        # elem_size = len(next(it))
        # if not all(len(elem) == elem_size for elem in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))
        obs_lens = torch.tensor([len(t) for t in transposed[2]])
        obs = torch.cat(transposed[2], dim=0)
        # padding obs_idxs here
        obs_idxs = list(transposed[3])
        max_len = max([len(obs_idx) for obs_idx in obs_idxs])
        
        for i in range(len(obs_idxs)):
            if len(obs_idxs[i]) < max_len:
                obs_idxs[i] += [obs_idxs[i][-1]] * (max_len - len(obs_idxs[i]))
        obs_idxs = tuple(obs_idxs)
       
        return [default_collate(transposed[0]), list(chain.from_iterable(transposed[1])), obs, obs_lens, obs_idxs] # task (B x 1), subgoals (B x 1), obs (total number of frames x C x H x W), obs_idxs (B x seq_len)
        
    else:  # Fall back to `default_collate`
        return default_collate(batch)

# Average metric meter
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
# save checkpoint
def save_checkpoint(state, is_best, filedir, filename='checkpoint.pth.tar'):
    torch.save(state, filedir / filename)
    shutil.copyfile(filedir / filename, filedir / 'checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filedir / filename, filedir / 'checkpoint_best.pth.tar')
