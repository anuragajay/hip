import itertools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from typing import Optional, Tuple

import clip
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModel

class ConsistencyClassifier(nn.Module):
    def __init__(self, img_feature_extractor, text_feature_extractor, classifier_arch, hidden_dims, output_dim, concat_before, dataset_type, task, device='cpu'):
        super().__init__()
        self.img_feature_extractor = img_feature_extractor
        self.text_feature_extractor = text_feature_extractor
        self.concat_before = concat_before
        self.device = device
        self.output_dim = output_dim
        self.dataset_type = dataset_type

        # if pretrained feature extractor is specified, then define both tokenizer & pretrained feature
        # extractor models
        if self.text_feature_extractor:
            if self.text_feature_extractor == 'clip':
                self.text_feat_ext, _ = clip.load('ViT-B/32', device=device)
                self.tokenizer = clip.tokenize
                self.text_feature_dim = 512
            elif self.text_feature_extractor == 'bert':
                self.text_feat_ext = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased') 
                self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
                self.text_feature_dim = 0 # TODO: fix feature dim accordingly
            # load flan-t5 huggingface
            elif self.text_feature_extractor == 'flan-t5':
                self.text_feat_ext = AutoModel.from_pretrained('google/flan-t5-base').to(device)
                self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
                self.text_feature_dim = 768
            elif self.text_feature_extractor == 'gpt-2':
                pass
            else:
                raise NotImplementedError(f'{self.text_feature_extractor} not implemented')
        
        if self.img_feature_extractor:
            if self.img_feature_extractor == 'clip':
                self.img_feat_ext = self.text_feat_ext if self.text_feature_extractor == 'clip' else clip.load('ViT-B/32', device=device)[0]
                self.img_feature_dim = 512
            elif self.img_feature_extractor == 'conv':
                self.img_feat_ext = ResNetSmall()
                # self.img_feat_ext = Net()
                self.img_feature_dim = 128
            elif self.img_feature_extractor == 'resnet18':
                self.img_feat_ext = torchvision.models.resnet18(pretrained=True)
                self.img_feat_ext = nn.Sequential(*list(self.img_feat_ext.children())[:-1])
                self.img_feature_dim = 512
            elif self.img_feature_extractor == 'resnet34':
                self.img_feat_ext = torchvision.models.resnet34(pretrained=True)
                self.img_feat_ext = nn.Sequential(*list(self.img_feat_ext.children())[:-1])
                self.img_feature_dim = 512
        

        # turn off gradients for encoder
        if self.text_feature_extractor:
            for param in self.text_feat_ext.parameters():
                param.requires_grad = False
        if self.img_feature_extractor == 'clip':
            for param in self.img_feat_ext.parameters():
                param.requires_grad = False
        
        if not self.concat_before:
            if self.dataset_type == 'all':
                if task == 'cliport':
                    self.feature_dim = self.img_feature_dim + self.text_feature_dim * 6 # 5 subgoals + 1 task + 1 obs
                elif task == 'paint':
                    self.feature_dim = self.img_feature_dim + self.text_feature_dim * 7 # 6 subgoals + 1 task + 1 obs
                else:
                    raise Exception(f'feature dim unspecified for task {task}')
            else:
                self.feature_dim = self.img_feature_dim + self.text_feature_dim * 2 # 1 subgoals + 1 task + 1 obs
        else:
            self.feature_dim = self.img_feature_dim + self.text_feature_dim * 1 # 1 subgoals and 1 task concatenated and fed to feature extractor

        # define classifier model
        if classifier_arch == 'mlp':
            layers = []
            layers.append(nn.Linear(self.feature_dim, hidden_dims[0]))
            layers.append(nn.ReLU())

            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
            self.clf = nn.Sequential(*layers)
        
    def forward(self, task, subgoals, obs, all_subgoals=False):
        if all_subgoals:
            subgoals = [s.split('\s') for s in subgoals]
            subgoals = list(itertools.chain.from_iterable(subgoals))

        # concat task & subgoals before encoder
        with torch.no_grad():
            if self.concat_before:
                x_text = torch.cat((task, subgoals), dim=1)

                if self.text_feature_extractor:
                    x_text = self.tokenizer(x_text).to(self.device)
                    x_text = self.text_feat_ext.encode_text(x).detach().float()
                    
            else:
                if self.text_feature_extractor:
                    if self.text_feature_extractor == 'flan-t5':
                        task = self.tokenizer(task, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.device)
                        task = self.text_feat_ext(task, decoder_input_ids=task).last_hidden_state.mean(dim=1).detach().float()

                        subgoals = self.tokenizer(subgoals, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.device)
                        subgoals = self.text_feat_ext(subgoals, decoder_input_ids=subgoals).last_hidden_state.mean(dim=1).detach().float()
                    elif self.text_feature_extractor == 'clip':
                        task = self.tokenizer(task, truncate=True).to(self.device)
                        task = self.text_feat_ext.encode_text(task).detach().float()

                        subgoals = self.tokenizer(subgoals, truncate=True).to(self.device)
                        subgoals = self.text_feat_ext.encode_text(subgoals).detach().float()
                        subgoals = subgoals.reshape(task.shape[0], -1) # B x number of subgoals (6)

                x_text = torch.cat((task, subgoals), dim=-1)

        if self.img_feature_extractor == 'clip':
            with torch.no_grad():
                x_img = self.img_feat_ext.encode_image(obs).detach().float()
        elif self.img_feature_extractor == 'conv':
            x_img = self.img_feat_ext(obs)
        elif 'resnet' in self.img_feature_extractor:
            x_img = self.img_feat_ext(obs).squeeze()
        
        x = torch.cat([x_text, x_img], dim=-1)
        logits = self.clf(x)

        return logits


class ConsistencyScorer(nn.Module):
    def __init__(self, vocab_size, img_feature_extractor=None, text_feature_extractor=None, scorer_arch='rnn', hidden_dims=None, dropout=0.0, concat_before=False, device='cpu'):
        super().__init__()
        self.img_feature_extractor = img_feature_extractor
        self.text_feature_extractor = text_feature_extractor
        self.scorer_arch = scorer_arch
        self.device = device
        self.hidden_dim = hidden_dims[0]
        self.output_dim = vocab_size
        self.dropout = dropout
        self.concat_before = concat_before

        # if pretrained feature extractor is specified, then define both tokenizer & pretrained feature
        # extractor models
        if self.text_feature_extractor:
            if self.text_feature_extractor == 'clip':
                self.text_feat_ext, _ = clip.load('ViT-B/32', device=device)
                self.tokenizer = clip.tokenize
                self.text_feature_dim = 512
            elif self.text_feature_extractor == 'bert':
                self.text_feat_ext = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased') 
                self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
                self.text_feature_dim = 0 # TODO: fix feature dim accordingly
            # load flan-t5 huggingface
            elif self.text_feature_extractor == 'flan-t5':
                self.text_feat_ext = AutoModel.from_pretrained('google/flan-t5-base').to(device)
                self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
                self.text_feature_dim = 768
            elif self.text_feature_extractor == 'gpt-2':
                pass
            else:
                raise NotImplementedError(f'{self.text_feature_extractor} not implemented')
        else: 
            self.text_feature_dim = vocab_size
        
        if self.img_feature_extractor:
            if self.img_feature_extractor == 'clip':
                self.img_feat_ext = self.text_feat_ext if self.text_feature_extractor == 'clip' else clip.load('ViT-B/32', device=device)[0]
                self.img_feature_dim = 512
        else: 
            raise NotImplementedError('No image feature extractor not implemented')
                    
        # turn off gradients for encoder
        if self.text_feature_extractor:
            for param in self.text_feat_ext.parameters():
                param.requires_grad = False
        if self.img_feature_extractor:
            for param in self.img_feat_ext.parameters():
                param.requires_grad = False

        self.feature_dim = self.img_feature_dim + self.text_feature_dim if self.concat_before else self.img_feature_dim + self.text_feature_dim * 2 # 1 img + 1 text (subgoal + task) or 1 img + 2 text (task + subgoal)


        # define scorer model
        # rnn scorer
        if scorer_arch == 'rnn':
            self.scorer = nn.RNN(input_size=self.feature_dim, hidden_size=self.hidden_dim, num_layers=len(hidden_dims), nonlinearity='relu', batch_first=True, dropout=self.dropout)
        
        # output layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, task, subgoal, obs, hidden):
        if hidden is None:
            hidden = self.init_hidden(len(task))
        
        # get text features from pretrained model if specified
        with torch.no_grad():
            if self.concat_before:
                x_text = torch.cat((task, subgoal), dim=1)

                if self.text_feature_extractor:
                    if self.text_feature_extractor == 'flan-t5':
                        x_text = self.tokenizer(x_text, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.device)
                        x_text = self.text_feat_ext(x_text, decoder_input_ids=x_text).last_hidden_state.mean(dim=1).detach().float()
                    else:
                        x_text = self.tokenizer(x_text, truncate=True).to(self.device)
                        x_text = self.text_feat_ext.encode_text(x).detach().float()
                    
            else:
                if self.text_feature_extractor:
                    if self.text_feature_extractor == 'flan-t5':
                        task = self.tokenizer(task, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.device)
                        task = self.text_feat_ext(task, decoder_input_ids=task).last_hidden_state.mean(dim=1).detach().float()

                        subgoal = self.tokenizer(subgoal, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(self.device)
                        subgoal = self.text_feat_ext(subgoal, decoder_input_ids=subgoal).last_hidden_state.mean(dim=1).detach().float()

                x_text = torch.cat((task, subgoal), dim=-1)

            if self.img_feature_extractor:
                x_img = self.img_feat_ext.encode_image(obs).detach().float()
        
        x = torch.cat([x_text, x_img], dim=-1)
        outputs, hidden = self.scorer(x)
        outputs = self.fc(outputs)

        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, device=self.device)

nonlinearity = nn.SiLU()

def video_avgpool_downsample(x, k=2):
    """Spatial downsampling via average pooling."""
    B, C, H, W = x.shape
    assert H % k == 0, "Maybe bug?"
    assert W % k == 0, "Maybe bug?"
    x = F.avg_pool2d(x, (k, k), stride=(k, k))
    assert x.shape == (B, C, H // k, W // k)
    return x


class ResnetBlock(nn.Module):
    """Convolutional residual block."""

    def __init__(self, 
                 kernel_size: Tuple[int, int, int],
                 out_ch: Optional[int] = None,
                 resample: Optional[str] = 'down',
                 ):
        super(ResnetBlock, self).__init__()
        self.kernel_size = kernel_size
        self.out_ch = out_ch
        self.resample = resample

        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=(1, 1))
        self.norm1 = nn.GroupNorm(1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resnet block.

        Args:
          x: A tensor of shape [B, C, H, W].

        Returns:
          The layers result.
        """
        B, C, _, _ = x.shape
        assert C == self.out_ch

        h = self.conv1(x)
        h = nonlinearity(self.norm1(h))

        if self.resample == 'down':
            h = video_avgpool_downsample(h)
            x = video_avgpool_downsample(x)

        h = self.conv2(h)

        return (x + h) / np.sqrt(2.0)

class ResNetSmall(nn.Module):
    """A inverse dynamic architecture."""

    def __init__(self, hidden_dim=128):
        super(ResNetSmall, self).__init__()

        self.conv_in = nn.Conv2d(3, hidden_dim, kernel_size=(3, 3), padding=(1, 1))

        self.down_blocks = nn.ModuleList([
            ResnetBlock(kernel_size=(3, 3), out_ch=hidden_dim, resample='down')
            for _ in range(3)
        ])

        self.action_1 = nn.Linear(hidden_dim, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse dynamics architecture.

        Args:
          x: Inputs of shape (B, C, H, W).

        Returns:
          Outputs.
        """

        B, C, H, W = x.shape
        assert x.dtype in (torch.float32, torch.float64)

        h = x
        h = self.conv_in(h)

        for down_block in self.down_blocks:
            h = down_block(h)

        h = h.mean(dim=-1).mean(dim=-1)
        h = nonlinearity(self.action_1(h))
        return h

class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.conv_res1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, stride=stride, bias=False)
        self.conv_res1_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
        self.conv_res2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding, bias=False)
        self.conv_res2_bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.9)

        if stride != 1:
            # in case stride is not set to 1, we need to downsample the residual so that
            # the dimensions are the same when we add them together
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels, momentum=0.9)
            )
        else:
            self.downsample = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv_res1_bn(self.conv_res1(x)))
        out = self.conv_res2_bn(self.conv_res2(out))

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.relu(out)
        out += residual
        return out


class Net(nn.Module):
    """
    A Residual network.
    """
    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 1024 feature dim

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        # out = self.fc(out)
        return out