import random
import warnings
import os
import getpass
import sys
import operator

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torchmetrics import Accuracy
from tqdm import tqdm
import wandb

from networks import ConsistencyClassifier, ConsistencyScorer
from datasets import create_dataloaders
from utils import process_hparams, AverageMeter, save_checkpoint, pad_list_of_strings, convert_token_to_label
from arguments import parse_arguments


def main():
    args = parse_arguments()
    print('args:', args, flush=True)

    if args.submit:
        make_sh_and_submit(args)
        return

    # set deterministic seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    # set up tensorboard logger and log hyperparameters
    if args.model_type == 'classifier':
        args.exp_id = f'task{args.task}_dataset{args.dataset_type}_sample{args.sample_ratio}_img{args.img_feature_extractor}_text{args.text_feature_extractor}_{args.classifier_arch}_{args.hidden_dims}_lr{args.lr}_scheduler{args.lr_scheduler}_bs{args.batch_size}_seed{args.seed}'
    elif args.model_type == 'scorer':
        args.exp_id = f'arch{args.scorer_arch}_img{args.img_feature_extractor}_text{args.text_feature_extractor}_hidden{args.hidden_dims}_{args.concat_before}_lr{args.lr}_scheduler{args.lr_scheduler}_bs{args.batch_size}_seed{args.seed}'
    args.log_dir = os.path.join(args.log_dir, args.exp_id)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp_id)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    wandb.init(project=f'llm_diffusion', name=args.exp_id, config=args, save_code=True)
    wandb.define_metric('train/step')
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric('epoch')
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("best/*", step_metric="epoch")
    
    # process hyperparameters
    args = process_hparams(args)

    # set up model
    print('=> creating model...', flush=True)
    if args.model_type == 'classifier':
        model = ConsistencyClassifier(args.img_feature_extractor, args.text_feature_extractor, args.classifier_arch, args.hidden_dims, args.output_dim, args.concat_before, args.dataset_type, args.task, args.gpu)
        train = train_clf
        validate = validate_clf
        metric_op = operator.gt # for higher is better

        # loss & metric
        if args.output_dim == 1:
            criterion = nn.BCEWithLogitsLoss().cuda()
            metric = Accuracy('binary').cuda()
        elif args.output_dim > 1:
            criterion = nn.CrossEntropyLoss().cuda()
            metric = Accuracy('multiclass', num_classes=args.output_dim).cuda()
    elif args.model_type == 'scorer':
        args.vocab_size += 3 # add <pad>, <bos>, <eos> token, but no gradients will be propagated along this label
        model = ConsistencyScorer(args.vocab_size, args.img_feature_extractor, args.text_feature_extractor, args.scorer_arch, args.hidden_dims, args.dropout, args.concat_before, args.gpu)
        train = train_scorer
        validate = validate_scorer
        metric_op = operator.lt # for lower is better

        # loss & metric
        criterion = nn.functional.cross_entropy
        
        metric = token_to_label = {} # including token_to_label for clarity of naming
        token_to_label['<pad>'] =  0
        token_to_label['<bos>'] =  1
        token_to_label['<eos>'] =  2

    model = model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, weight_decay=args.weight_decay)

    # load checkpoint
    if os.path.isfile(args.checkpoint_dir / 'checkpoint.pth.tar'):
        print("=> loading checkpoint '{}'".format(args.checkpoint_dir / 'checkpoint.pth.tar'))
        checkpoint = torch.load(args.checkpoint_dir / 'checkpoint.pth.tar', map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
        best_epoch = checkpoint['best_epoch']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.checkpoint_dir / 'checkpoint.pth.tar', checkpoint['epoch']))
        
        if args.start_epoch >= args.epochs:
            print('=> already trained for {} epochs'.format(args.epochs))
            return
    else:
        start_epoch = 0
        best_metric = 0. if metric_op == operator.gt else float('inf')
    
    # dataloader
    print('=> creating dataloader...', flush=True)
    train_loader, test_loader = create_dataloaders(args)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) if args.lr_scheduler == 'cosine' else None

    best_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        train(model, optimizer, train_loader, scheduler, criterion, metric, epoch, args)
        val_metric = validate(model, test_loader, criterion, metric, epoch, args)

        is_best = metric_op(val_metric, best_metric)
        best_epoch = epoch if is_best else best_epoch
        best_metric = val_metric if is_best else best_metric

        if (epoch+1) % args.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_metric': best_metric,
                'best_epoch': best_epoch,
                'optimizer' : optimizer.state_dict(),
                'args': vars(args),
            }, is_best, args.checkpoint_dir, f'checkpoint_{epoch}.pth.tar')

        wandb.log({'best/best_metric': best_metric, 'best/best_epoch': best_epoch, 'epoch': epoch})

# train function for rnn
def train_scorer(model, optimizer, train_loader, scheduler, criterion, token_to_label, epoch, args):
    print(f'=> training epoch {epoch}...', flush=True)
    model.train()

    for step, (task, subgoals, obs, obs_lens, obs_idxs) in enumerate(tqdm(train_loader), start=epoch * len(train_loader)):
        # add bos token to beginning of subgoals

        if torch.cuda.is_available():
            # task = task.cuda(non_blocking=True)
            # subgoals = subgoals.cuda(non_blocking=True)
            obs = obs.cuda(non_blocking=True)

        # initialize hidden state
        hidden = None

        subgoals = [s.split() for s in subgoals] # split subgoals into list of tokens
        subgoals = pad_list_of_strings(subgoals, obs_idxs) # pad with <pad> to max seq len
        seq_len = len(subgoals[0]) 
        subgoals = list(zip(*subgoals)) # transpose to list of seq_len x batch_size
        obs_idxs = torch.tensor(list(zip(*obs_idxs))) # transpose to list of seq_len x batch_size

        # using cumulative index calculated from obs_lens and relative idx from obs_idxs to get actual idx
        cum_idxs = torch.cumsum(obs_lens, dim=0)
        cum_idxs[1:] = cum_idxs[:-1].clone()
        cum_idxs[0] = 0

        loss = 0.0
        for s in range(seq_len-1):
            cur_subgoals = list(zip(*subgoals[:s+1]))
            cur_subgoals = [' '.join(c_s) for c_s in cur_subgoals]
            next_subgoals = list(subgoals[s+1])
            next_subgoals = torch.tensor(convert_token_to_label(next_subgoals, token_to_label)).long().cuda(non_blocking=True) # convert on the fly
            # for loop over subgoals
            cur_obs_idxs = cum_idxs + obs_idxs[s]
            outputs, hidden = model(task, cur_subgoals, obs[cur_obs_idxs], hidden)
            loss += criterion(outputs, next_subgoals, ignore_index=token_to_label['<pad>']) # ignore the label for <pad> token

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_freq == 0:
            pp = np.exp(loss.item() / seq_len)
            print(f'epoch={epoch} step={step} loss={loss.item():.4f} perplexity={pp:.2f}', flush=True)
            wandb.log({'train/loss': loss.item(), 'train/perplexity': pp, 'train/step': step})

        if scheduler is not None:
            scheduler.step()

def train_clf(model, optimizer, train_loader, scheduler, criterion, metric, epoch, args):
    print(f'=> training epoch {epoch}...', flush=True)
    model.train()

    for step, (task, subgoals, obs, label) in enumerate(tqdm(train_loader), start=epoch * len(train_loader)):
        if torch.cuda.is_available():
            # task = task.cuda(non_blocking=True)
            # subgoals = subgoals.cuda(non_blocking=True)
            obs = obs.cuda(non_blocking=True)
            label = label.view(-1, 1).cuda(non_blocking=True)

        output = model(task, subgoals, obs, all_subgoals=True if args.dataset_type == 'all' else False)
        loss = criterion(output, label) if args.output_dim == 1 else criterion(output, label.squeeze().long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.print_freq == 0:
            if args.output_dim == 1:
                acc = metric(torch.sigmoid(output), label)
            else:
                acc = metric(output.argmax(dim=-1), label.squeeze())
            print(f'epoch={epoch}, step={step}, loss={loss.item()}, acc={acc.item()}', flush=True)

            log_dict = {
                'train/loss': loss.item(),
                'train/acc': acc.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
            }

            wandb.log(log_dict)

    if scheduler is not None:
        scheduler.step()


def validate_clf(model, test_loader, criterion, metric, epoch, args):
    print(f'=> validating epoch {epoch}...', flush=True)
    model.eval()

    val_loss = AverageMeter('ValLoss')
    val_acc = AverageMeter('ValAcc')

    with torch.no_grad():
        for step, (task, subgoals, obs, label) in enumerate(tqdm(test_loader), start=epoch * len(test_loader)):
            if torch.cuda.is_available():
                # task = task.cuda(non_blocking=True)
                # subgoals = subgoals.cuda(non_blocking=True)
                obs = obs.cuda(non_blocking=True)
                label = label.view(-1, 1).cuda(non_blocking=True)
            output = model(task, subgoals, obs, all_subgoals=True if args.dataset_type == 'all' else False)
            loss = criterion(output, label) if args.output_dim == 1 else criterion(output, label.squeeze().long())
            if args.output_dim == 1:
                acc = metric(torch.sigmoid(output), label)
            else:
                acc = metric(output.argmax(dim=-1), label.squeeze())
            
            val_acc.update(acc.item(), label.size(0))
            val_loss.update(loss.item(), label.size(0))

    # log to wandb
    log_dict = {
        'val/loss': val_loss.avg,
        'val/acc': val_acc.avg,
        'epoch': epoch
    }

    wandb.log(log_dict)
    print(f'val epoch={epoch}, loss={val_loss.avg}, acc={val_acc.avg}', flush=True)
    
    return val_acc.avg

def validate_scorer(model, test_loader, criterion, token_to_label, epoch, args):
    print(f'=> validating epoch {epoch}...', flush=True)
    model.eval()

    val_loss = AverageMeter('ValLoss')
    val_pp = AverageMeter('ValPerplexity')

    with torch.no_grad():
        for step, (task, subgoals, obs, obs_lens, obs_idxs) in enumerate(tqdm(test_loader), start=epoch * len(test_loader)):
            if torch.cuda.is_available():
                obs = obs.cuda(non_blocking=True)

            # initialize hidden state
            hidden = None

            subgoals = [s.split() for s in subgoals] # split subgoals into list of tokens
            subgoals = pad_list_of_strings(subgoals, obs_idxs) # pad with <pad> to max seq len
            seq_len = len(subgoals[0]) 
            subgoals = list(zip(*subgoals)) # transpose to list of seq_len x batch_size
            obs_idxs = torch.tensor(list(zip(*obs_idxs))) # transpose to list of seq_len x batch_size

            # using cumulative index calculated from obs_lens and relative idx from obs_idxs to get actual idx
            cum_idxs = torch.cumsum(obs_lens, dim=0)
            cum_idxs[1:] = cum_idxs[:-1].clone()
            cum_idxs[0] = 0

            loss = 0.0
            for s in range(seq_len-1):
                cur_subgoals = list(zip(*subgoals[:s+1]))
                cur_subgoals = [' '.join(c_s) for c_s in cur_subgoals]
                next_subgoals = list(subgoals[s+1])
                next_subgoals = torch.tensor(convert_token_to_label(next_subgoals, token_to_label)).long().cuda(non_blocking=True) # convert on the fly
                # for loop over subgoals
                cur_obs_idxs = cum_idxs + obs_idxs[s]
                outputs, hidden = model(task, cur_subgoals, obs[cur_obs_idxs], hidden)
                loss += criterion(outputs, next_subgoals, ignore_index=token_to_label['<pad>']) # ignore the label for <pad> token

            if step % args.print_freq == 0:
                pp = np.exp(loss.item() / seq_len)
                print(f'val epoch={epoch} step={step} loss={loss.item():.4f} perplexity={pp:.2f}', flush=True)
                # wandb.log({'val/loss': loss.item(), 'val/perplexity': pp})
            
            val_loss.update(loss.item(), len(task))
            val_pp.update(pp, len(task))

    # log to wandb
    log_dict = {
        'val/loss': val_loss.avg,
        'val/perpelxity': val_pp.avg,
    }    
    wandb.log(log_dict)

    return val_pp.avg




def make_sh_and_submit(args, delay=0):
    os.makedirs('./scripts/submit_scripts/', exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    options = args.arg_str
    
    if delay == 0:
        options_split = options.split(" ")
        name = ''.join([opt1.replace("--","").replace("=","").replace('gpu', '').replace('print-freq', '') for opt1 in options_split])
        name = args.add_prefix + name

    else: # log_id should be already defined
        name = args.log_id
    print('Submitting the job with options: ')
    # print(options)
    print(f"experiment name: {name}")

    if args.server == 'insert server name':
        options += f' --server=<insert server name> --arg_str=\"{args.arg_str}\" '
        preamble = (
            f'#!/bin/sh\n#SBATCH --gres=gpu:2\n#SBATCH --exclusive\n#SBATCH --cpus-per-task=20\n#SBATCH '
            f'-N 1\n#SBATCH -t 360\n#SBATCH ')
        preamble += f'--begin=now+{delay}hour\n#SBATCH '
        preamble += (f'-o ./logs/{name}.out\n#SBATCH '
                        f'--job-name={name}_{delay}\n#SBATCH '
                        f'--open-mode=append\n\n')

    else:
        username = getpass.getuser()
        options += f' --server={args.server} '
        preamble = (
            f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH '
            f'-o ./logs/{name}.out\n#SBATCH '
            f'--job-name={name}\n#SBATCH '
            f'--open-mode=append\n\n'
        )
    with open(f'./scripts/submit_scripts/{name}_{delay}.sh', 'w') as file:
        file.write(preamble)
        file.write("echo \"current time: $(date)\";\n")
        port = random.randrange(10000, 20000)
        file.write(f'wandb offline\n')
        file.write(
            f'python {sys.argv[0]} {options} '
        )

        if args.server == 'sc':
            if args.task == 'paint':
                file.write(f'--data /path/to/data')
            elif args.task == 'cliport':
                file.write(f'--data /path/to/data')

    os.system(f'sbatch ./scripts/submit_scripts/{name}_{delay}.sh')

if __name__ == '__main__':
    main()
