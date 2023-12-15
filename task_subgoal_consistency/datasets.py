import random
import os
import pickle
import glob
import itertools

import torch
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision import transforms
from PIL import Image

from utils import custom_collate_clf, custom_collate_scorer, get_goal


# restructured data format
class TaskSubgoalsObsNewDataset(Dataset):
    def __init__(self, data_dir, img_transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.img_transforms = img_transforms
        self.file_list = {}

        if os.path.isfile('./misc/task_subgoals_obs_dataset_files.pkl'):
            print('loading pre-computed file list', flush=True)
            with open('./misc/task_subgoals_obs_dataset_files.pkl', 'rb') as f:
                # self.file_list = [os.path.join(self.data_dir, line.strip()) for line in f.readlines()]
                self.file_list = pickle.load(f)
        else:
            print('computing file list', flush=True)
            all_tasks = os.listdir(self.data_dir)
            for i in range(len(all_tasks)):
                traj_paths = sorted(os.listdir(os.path.join(self.data_dir, all_tasks[i])))
                file_dict = {'trajs': [os.path.join(self.data_dir, all_tasks[i], t) for t in traj_paths]}

                for t in file_dict['trajs']:
                    imgs_paths = sorted(glob.glob(os.path.join(t, '*.png')))
                    file_dict[t] = imgs_paths

                self.file_list[i] = file_dict

            with open('./misc/task_subgoals_obs_dataset_files.pkl', 'wb') as f:
                pickle.dump(self.file_list, f)

    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        trajs = self.file_list[idx]['trajs']
        metadatas = [torch.load(os.path.join(t, 'metadata.pt')) for t in trajs]
    
        # load task and sanity checking that task is consistent
        task = metadatas[0]['task']

        # load all subgoals
        subgoals = [m['subgoal'] for m in metadatas]
        
        # get subgoal lengths to get index of the corresponding obs
        subgoals_split = [s.split() for s in subgoals]
        subgoals_lens = [len(s) for s in subgoals_split]
        obs_idxs = [[i] * subgoals_lens[i] for i in range(len(subgoals_lens))]
        obs_idxs = list(itertools.chain.from_iterable(obs_idxs))
        obs_idxs = [0] + obs_idxs + [len(subgoals_lens)-1] # <bos> gets initial state and <eos> tokens get last state obs

        subgoals = ['<bos>'] + subgoals + ['<eos>'] # attaching <bos> token to the start of subgoals
        subgoals = [' '.join(subgoals)]

        # load a random observation from each trajectory
        obs = [Image.open(random.choice(self.file_list[idx][trajs[i]])) for i in range(len(trajs))]
        
        # apply image transforms onto observation if provided
        if self.img_transforms:
            obs = torch.stack([self.img_transforms(o) for o in obs], dim=0)

        return task, subgoals, obs, obs_idxs
    
# restructured data format
class TaskSubgoalsObsClfDataset(Dataset):
    def __init__(self, data_dir, task, img_transforms=None, negative_sample_prob=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.task = task
        self.img_transforms = img_transforms
        self.file_list = {}
        self.negative_sample_prob = negative_sample_prob
        self.task = task
        filename = f'./misc/{task}_task_subgoals_obs_dataset_files.pkl'
        
        if os.path.isfile(filename):
            print('loading pre-computed file list', flush=True)
            with open(filename, 'rb') as f:
                self.file_list = pickle.load(f)
        else:
            print('computing file list', flush=True)
            all_tasks = os.listdir(self.data_dir)
            count = 0
            for i in range(len(all_tasks)):
                traj_paths = sorted(os.listdir(os.path.join(self.data_dir, all_tasks[i])))
                file_dict = {'trajs': [os.path.join(self.data_dir, all_tasks[i], t) for t in traj_paths]}

                missing_md = False
                for t in file_dict['trajs']:
                    imgs_paths = sorted(glob.glob(os.path.join(t, '*.png')))
                    file_dict[t] = imgs_paths

                    if not os.path.isfile(os.path.join(t, 'metadata.pt')):
                        missing_md = True
                        break
                
                if missing_md:
                    continue

                self.file_list[count] = file_dict
                count += 1

            with open(filename, 'wb') as f:
                pickle.dump(self.file_list, f)

    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        trajs = self.file_list[idx]['trajs']
        subgoal_idx = random.randint(0, len(trajs)-1)

        metadatas = [torch.load(os.path.join(t, 'metadata.pt')) for t in trajs]
        subgoals = [m['subgoal'] for m in metadatas]
        if self.task == 'paint':
            task = get_goal(subgoals)
        else:
            task = metadatas[0]['task']
        subgoal = subgoals[subgoal_idx]

        # randomly sample a negative sample
        if random.random() < self.negative_sample_prob and len(trajs) > 1:
            label = torch.zeros(1) # negative sample
            obs_idx = random.choice([i for i in range(0, len(trajs)) if i not in [subgoal_idx]])
            obs_subgoal = subgoals[obs_idx]
            if obs_subgoal == subgoal:
                obs_idx = subgoal_idx
                label = torch.ones(1)
                
        # positive sample
        else: 
            label = torch.ones(1) 
            obs_idx = subgoal_idx
        
        obs = Image.open(random.choice(self.file_list[idx][trajs[obs_idx]])).convert('RGB')

        # apply image transforms onto observation if provided
        if self.img_transforms:
            obs = self.img_transforms(obs)

        return task, subgoal, obs, label

# restructured data format
class TaskSubgoalsObsClfSubsetDataset(Dataset):
    def __init__(self, data_dir, img_transforms=None, negative_sample_prob=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.img_transforms = img_transforms
        self.file_list = {}
        self.negative_sample_prob = negative_sample_prob
        
        if os.path.isfile('./misc/task_subgoals_obs_dataset_files.pkl'):
            print('loading pre-computed file list', flush=True)
            with open('./misc/task_subgoals_obs_dataset_files.pkl', 'rb') as f:
                # self.file_list = [os.path.join(self.data_dir, line.strip()) for line in f.readlines()]
                self.file_list = pickle.load(f)
        else:
            print('computing file list', flush=True)
            all_tasks = os.listdir(self.data_dir)
            for i in range(len(all_tasks)):
                traj_paths = sorted(os.listdir(os.path.join(self.data_dir, all_tasks[i])))
                file_dict = {'trajs': [os.path.join(self.data_dir, all_tasks[i], t) for t in traj_paths]}

                for t in file_dict['trajs']:
                    imgs_paths = sorted(glob.glob(os.path.join(t, '*.png')))
                    file_dict[t] = imgs_paths

                self.file_list[i] = file_dict

            with open('./misc/task_subgoals_obs_dataset_files.pkl', 'wb') as f:
                pickle.dump(self.file_list, f)

    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        trajs = self.file_list[idx]['trajs']
        subgoal_idx = random.randint(0, len(trajs)-1)

        metadatas = [torch.load(os.path.join(t, 'metadata.pt')) for t in trajs]
        subgoals = [m['subgoal'] for m in metadatas]
        task = get_goal(subgoals)
        subgoal_subset = subgoals[subgoal_idx:]
        subgoal = subgoals[subgoal_idx:]

        # randomly sample a negative sample
        if random.random() < self.negative_sample_prob and len(trajs) > 1:
            label = torch.zeros(1) # negative sample
            obs_idx = random.choice([i for i in range(0, len(trajs)) if i not in [subgoal_idx]])
            obs_subgoal = subgoals[obs_idx]
            if obs_subgoal == subgoal:
                obs_idx = subgoal_idx
                label = torch.ones(1)
                
        # positive sample
        else: 
            label = torch.ones(1) 
            obs_idx = subgoal_idx
        
        obs = Image.open(random.choice(self.file_list[idx][trajs[obs_idx]])).convert('RGB')

        # apply image transforms onto observation if provided
        if self.img_transforms:
            obs = self.img_transforms(obs)

        # join all strings in list
        subgoal_subset = ' '.join(subgoal_subset)

        return task, subgoal_subset, obs, label

# returns all subgoals in a trajectory and multi-class labels
class TaskSubgoalsObsClfAllDataset(Dataset):
    def __init__(self, data_dir, task, img_transforms=None, negative_sample_prob=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.img_transforms = img_transforms
        self.file_list = {}
        self.negative_sample_prob = negative_sample_prob
        self.task = task
        max_num_subgoals = 6 if task == 'paint' else 5
        filename = f'./misc/{task}_task_subgoals_obs_{max_num_subgoals}_dataset_files.pkl'

        if os.path.isfile(filename):
            print('loading pre-computed file list', flush=True)
            with open(filename, 'rb') as f:
                self.file_list = pickle.load(f)
        else:
            print('computing file list', flush=True)
            all_tasks = os.listdir(self.data_dir)
            count = 0
            for i in range(len(all_tasks)):
                traj_paths = sorted(os.listdir(os.path.join(self.data_dir, all_tasks[i])))
                if len(traj_paths) != max_num_subgoals:
                    continue

                file_dict = {'trajs': [os.path.join(self.data_dir, all_tasks[i], t) for t in traj_paths]}

                missing_md = False
                for t in file_dict['trajs']:
                    imgs_paths = sorted(glob.glob(os.path.join(t, '*.png')))
                    file_dict[t] = imgs_paths

                    if not os.path.isfile(os.path.join(t, 'metadata.pt')):
                        missing_md = True
                        break
                
                if missing_md:
                    continue

                self.file_list[count] = file_dict
                count += 1

            with open(filename, 'wb') as f:
                pickle.dump(self.file_list, f)

    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        trajs = self.file_list[idx]['trajs']

        metadatas = [torch.load(os.path.join(t, 'metadata.pt')) for t in trajs]
        subgoals = [m['subgoal'] for m in metadatas]
        subgoals = '\s'.join(subgoals) # join all subgoals with delimiter \s
        if self.task == 'paint':
            task = get_goal(subgoals)
        else:
            task = metadatas[0]['task']

        obs_idx = random.randint(0, len(trajs)-1)
        obs = Image.open(random.choice(self.file_list[idx][trajs[obs_idx]])).convert('RGB')

        # apply image transforms onto observation if provided
        if self.img_transforms:
            obs = self.img_transforms(obs)

        return task, subgoals, obs, obs_idx

class TaskSubgoalsObsDataset(Dataset):
    def __init__(self, data_dir, img_transforms=None, negative_sample_prob=0.5):
        super().__init__()
        self.data_dir = data_dir
        self.img_transforms = img_transforms

        if os.path.isfile('./misc/task_subgoals_obs_dataset_files.pkl'):
            print('loading pre-computed file list', flush=True)
            with open('./misc/task_subgoals_obs_dataset_files.pkl', 'rb') as f:
                # self.file_list = [os.path.join(self.data_dir, line.strip()) for line in f.readlines()]
                self.file_list = pickle.load(f)
        else:
            print('computing file list', flush=True)
            all_files = os.listdir(self.data_dir)
            self.file_list = [os.path.join(self.data_dir, f) for f in all_files if os.path.isfile(os.path.join(self.data_dir, f))]
            with open('./misc/task_subgoals_obs_dataset_files.pkl', 'wb') as f:
                pickle.dump(self.file_list, f)

        self.negative_sample_prob = negative_sample_prob

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx])
        task = data[0]['task']
        subgoals = [d['subgoal'] for d in data]

        # preprocessing text to remove underscores and strip
        task = task.replace('_', ' ').strip()
        subgoals = [subgoal.replace('_', ' ').strip() for subgoal in subgoals]
        # remove duplicates
        # subgoals = list(OrderedDict.fromkeys(subgoals))
        label = torch.ones(1) # positive sample
        obs = data[0]['obs']

        # with negative_sample_prob probability, return a negative sample
        if random.random() < self.negative_sample_prob:
            label = torch.zeros(1) # negative sample

            p = random.random()
            # permute the subgoals
            if p < 0.33:
                random.shuffle(subgoals)
            # use observation from a different subgoal
            elif p < 0.66:
                obs = random.choice(data[1:])['obs']
            # load another trajectory and swap subgoals
            else:
                data2 = torch.load(random.choice(self.file_list))
                subgoals2 = [d['subgoal'] for d in data2]
                subgoals2 = [subgoal.replace('_', ' ').strip() for subgoal in subgoals2]
                # subgoals2 = list(OrderedDict.fromkeys(subgoals2))
                swap_idx = random.randint(0, len(subgoals2)-1)
                subgoals = subgoals[:swap_idx] + subgoals2[swap_idx:]
        
        # apply image transforms onto observation if provided
        if self.img_transforms:
            obs = self.img_transforms(obs)

        return task, subgoals, obs, label

class TaskSubgoalsDataset(Dataset):
    def __init__(self, data_dir, negative_sample_prob=0.5):
        super().__init__()

        self.data_dir = data_dir

        if os.path.isfile('./misc/task_subgoals_dataset_files.pkl'):
            print('loading pre-computed file list', flush=True)
            with open('./misc/task_subgoals_dataset_files.pkl', 'rb') as f:
                # self.file_list = [os.path.join(self.data_dir, line.strip()) for line in f.readlines()]
                self.file_list = pickle.load(f)
        else:
            print('computing file list', flush=True)
            all_files = os.listdir(self.data_dir)
            self.file_list = [os.path.join(self.data_dir, f) for f in all_files if os.path.isfile(os.path.join(self.data_dir, f))]
            with open('./misc/task_subgoals_dataset_files.pkl', 'wb') as f:
                pickle.dump(self.file_list, f)

        self.negative_sample_prob = negative_sample_prob

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx])
        task = data['task']
        subgoals = data['subgoals']

        # preprocessing text to remove underscores and strip
        task = task.replace('_', ' ').strip()
        subgoals = [subgoal.replace('_', ' ').strip() for subgoal in subgoals]
        # remove duplicates
        # subgoals = list(OrderedDict.fromkeys(subgoals))
        label = torch.ones(1) # positive sample

        # with negative_sample_prob probability, return a negative sample
        if random.random() < self.negative_sample_prob:
            label = torch.zeros(1) # negative sample

            # permute the subgoals
            if random.random() < 0.5:
                random.shuffle(subgoals)

            # load another trajectory and swap subgoals
            else:
                data2 = torch.load(random.choice(self.file_list))
                subgoals2 = data2['subgoals']
                subgoals2 = [subgoal.replace('_', ' ').strip() for subgoal in subgoals2]
                # subgoals2 = list(OrderedDict.fromkeys(subgoals2))
                swap_idx = random.randint(0, len(subgoals2)-1)
                subgoals = subgoals[:swap_idx] + subgoals2[swap_idx:]

        # subgoals = ' '.join(subgoals)

        return task, subgoals, label


# create dataloader for training
def create_dataloaders(args):
    # dataset = TaskSubgoalsDataset(args.data, args.negative_sample_prob)

    # define defalt image transforms
    if args.img_feature_extractor == 'clip':
        import clip
        img_transforms = clip.load('ViT-B/32', device='cpu')[1]
        # img_transforms.transforms.insert(0,transforms.ToPILImage())
    else:
        img_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if args.model_type == 'scorer':
        dataset = TaskSubgoalsObsNewDataset(args.data, img_transforms)
        custom_collate = custom_collate_scorer
    elif args.model_type == 'classifier':
        if args.dataset_type == 'single':
            dataset = TaskSubgoalsObsClfDataset(args.data, args.task, img_transforms, args.negative_sample_prob)
        elif args.dataset_type == 'subset': 
            dataset = TaskSubgoalsObsClfSubsetDataset(args.data, args.task, img_transforms, args.negative_sample_prob)
        elif args.dataset_type == 'all':
            dataset = TaskSubgoalsObsClfAllDataset(args.data, args.task, img_transforms, args.negative_sample_prob)
        custom_collate = custom_collate_clf

    # split into train and val
    train_size = int(args.train_ratio * len(dataset))
    all_idxs = list(range(len(dataset)))
    train_idxs = random.sample(all_idxs, train_size)
    val_idxs = list(set(all_idxs) - set(train_idxs))

    if args.sample_ratio < 1.0:
        train_size = int(train_size * args.sample_ratio)
        train_idxs = random.sample(train_idxs, train_size)

    train_dataset = Subset(dataset, train_idxs)
    val_dataset = Subset(dataset, val_idxs)
    print('train size ', len(train_dataset))
    print('val size ', len(val_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate, pin_memory=False, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=custom_collate, pin_memory=False, num_workers=args.workers)

    return train_loader, val_loader
