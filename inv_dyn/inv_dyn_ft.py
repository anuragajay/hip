import threading
from io import BytesIO
from queue import Queue
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from ml_collections import ConfigDict
from PIL import Image
from scipy.spatial.transform import Rotation
import torchvision.transforms as transforms
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTConfig, ViTModel, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import AutoFeatureExtractor, ResNetForImageClassification, ResNetConfig, ResNetModel
from transformers import ViTFeatureExtractor, ViTModel
from transformers.image_utils import ChannelDimension
from inv_dynamics.action_decoder import ActionDecoder, MultiCategoricalNet, CategoricalNet
from inv_dynamics.dists import Categorical, MultiCategorical
from matplotlib import pyplot as plt
from tqdm import tqdm
from .resnet import ResNetSmall
from ffcv.fields.decoders import  NDArrayDecoder
from ffcv.transforms import ToTensor, Squeeze, ToDevice
from ffcv.loader import Loader, OrderOption
import vc_models
from vc_models.models.vit import model_utils

class InvDynamics(nn.Module):
    def __init__(self, state_dim=7):
        super(InvDynamics, self).__init__()
        self.state_dim = state_dim

        self.visual_model, self.embd_size, self.model_transforms, self.model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)
        
        self.inv_model = nn.Linear(self.embd_size, self.state_dim)
    
    def forward(self, obs):        
        # Get Action
        obs = self.model_transforms(obs)
        embed = self.visual_model(obs)
        return self.inv_model(embed)
    
    def calculate_loss(self, obs, state):
        pred_state = self.forward(obs)
        mse = F.mse_loss(pred_state, state)
        return mse

    @torch.no_grad()
    def calculate_test_loss(self, obs, state):
        pred_state = self.forward(obs)
        mse = F.mse_loss(pred_state, state)
        return mse


def main(**deps):
    from tqdm import tqdm
    import wandb
    import numpy as np
    import torch
    import random
    import os

    device=torch.device('cuda:0')
    log_every=25
    
    # Create Loaders
    wandb.init(
        project='llm_diffusion', 
        config={"lr": 3e-5, "batch_size":256},
        group='inv_model_ft',
    )

    batch_size = wandb.config.batch_size
    num_workers = 20
    parent_path = '/path/to/data'

    train_dataloader = Loader(f'{parent_path}/inv_dyn_train.beton', batch_size=batch_size,
                        num_workers=num_workers, order=OrderOption.RANDOM,
                        pipelines={
                        'image': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                        'state': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                    })

    test_dataloader = Loader(f'{parent_path}/inv_dyn_test.beton', batch_size=batch_size,
                        num_workers=num_workers, order=OrderOption.RANDOM,
                        pipelines={
                        'image': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                        'state': [NDArrayDecoder(), ToTensor(), ToDevice(device)],
                    })

    train_steps_per_epoch = len(train_dataloader)
    num_epochs = 20 

    # Define model and optimizer
    inv_model = InvDynamics()
    inv_model = inv_model.to(device)

    optimizer = torch.optim.AdamW(inv_model.parameters(), lr=wandb.config.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=5e-4)

    # Run optimization
    for epoch_num in tqdm(range(num_epochs)):
        # Run train epoch
        inv_model.train()
        running_training_loss = 0
        for (idx, (obs, state)) in enumerate(train_dataloader):
            obs = (obs.permute(0,3,1,2))/255.0
            obs = obs.contiguous()
            loss = inv_model.calculate_loss(obs, state)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(inv_model.parameters(), 1.0)
            optimizer.step()
            running_training_loss += loss.detach().item()
            del obs, state
            if idx > 0 and idx % log_every == 0:
                wandb.log({"epoch_num": epoch_num, "train_loss": running_training_loss/log_every, "itr": idx + epoch_num*train_steps_per_epoch})
                print({"epoch_num": epoch_num, "train_loss": running_training_loss/log_every, "itr": idx + epoch_num*train_steps_per_epoch})
                running_training_loss = 0

        # Run test epoch
        inv_model.eval()
        running_test_loss = 0
        with torch.no_grad():
            for (idx, (obs, state)) in enumerate(test_dataloader):
                obs = (obs.permute(0,3,1,2))/255.0
                obs = obs.contiguous()
                sin_cos_mse = inv_model.calculate_test_loss(obs, state)
                running_test_loss += sin_cos_mse
                del obs, state
        
        wandb.log({"epoch_num": epoch_num, "test_loss": running_test_loss/(idx+1), "itr": (epoch_num+1)*train_steps_per_epoch})
        print({"epoch_num": epoch_num, "test_loss": running_test_loss/(idx+1), "itr": (epoch_num+1)*train_steps_per_epoch})

        # Save the current progress
        torch.save({'inv_model':inv_model.state_dict(), 'opt': optimizer.state_dict()}, 'inv.pt')
    
    wandb.finish()