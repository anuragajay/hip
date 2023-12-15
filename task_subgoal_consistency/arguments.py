import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a classifier to check task subgoal consistency')
    parser.add_argument('--data', type=str, default='/path/to/data/', metavar='DIR', help='path to dataset')
    parser.add_argument('--log-dir', type=str, default='./logs/', help='path to log directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/', help='path to log directory')

    # model hyperparams
    parser.add_argument('--model-type', default='classifier', type=str, choices=['scorer', 'classifier'],
                         help='type of model to train')
    parser.add_argument('--img-feature-extractor', default='resnet18', type=str,
                        choices=('clip', 'conv', 'resnet18', 'resnet34'),
                        help='pretrained image feature extractor to use (default=None=> no extractor)')
    parser.add_argument('--text-feature-extractor', default='flan-t5', type=str,
                        choices=('clip', 'bert', 'gpt-2', 'flan-t5'),
                        help='pretrained text feature extractor to use (default=None=> no extractor)')
    parser.add_argument('--classifier-arch', default='mlp', type=str,
                        help='classifier architecture')
    parser.add_argument('--hidden-dims', help='hidden dimensions (in str) of the classifier delimited by comma', type=str, default='512,256,128')
    parser.add_argument('--output-dim', type=int, default=1, help='output dimension of the classifier')
    parser.add_argument('--vocab-size', type=int, default=22,
                        help='vocab size of the data')
    parser.add_argument('--embedding-dim', type=int, default=128,
                        help='embedding dimension for scorer') # maybe smaller?
    parser.add_argument('--scorer-arch', default='rnn', type=str, choices=['rnn', 'transformer'],
                        help='scorer architecture')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout applied to outputs of each rnn layer')

    # dataset hyperparameters
    parser.add_argument('--dataset-type', default='single', choices=['single', 'subset', 'all'], type=str,
                        help='whether to use single subgoal classification or subset classification')
    parser.add_argument('--task', default='paint', choices=['paint', 'cliport'], type=str,
                    help='which task to run on')
    parser.add_argument('--train-ratio', default=0.9, type=float,
                        help='ratio of data to use for training vs validation')
    parser.add_argument('--sample-ratio', default=1.0, type=float,
                        help='sample complexity ratio (proportion of training data to use from entire training data)')
    parser.add_argument('--negative-sample-prob', default=0.5, type=float,
                        help='probabiilty of sampling a negative example')

    # training hyperparameters
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', default=0.0, type=float, metavar='LR',
                        help='base learning rate')
    parser.add_argument('--lr-scheduler', default=None, type=str, metavar='LR-SCH',
                        choices=('cosine'),
                        help='scheduler for learning rate')
    parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--val-batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size for validation (uses only 1 gpu)')
    parser.add_argument('--concat-before', default=False, action='store_true',
                        help='whether to concat the task & subgoals before passing to encoders')

    # training environment configs
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loader workers')

    # submit configs
    parser.add_argument('--server', type=str, default='sc')
    parser.add_argument('--arg_str', default='--', type=str)
    parser.add_argument('--add_prefix', default='', type=str)
    parser.add_argument('--submit', action='store_true', default=False)

    # misc configs
    parser.add_argument('--gpu', default=-1, type=int, metavar='G',
                        help='gpu to use (default: -1 => use cpu)')
    parser.add_argument('--seed', default=None, type=int, metavar='S',
                        help='random seed')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                        help='print frequency in # of iterations')
    parser.add_argument('--save-freq', default=5, type=int, metavar='N',
                        help='save frequency in # of epochs')
    
    args = parser.parse_args()

    return args
    
