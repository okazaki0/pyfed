import os
import pickle
import re
import collections
from PIL import Image
from torch.utils.data import Dataset
import json
import torch
import logging
import string
import syft as sy  # <-- NEW: import the Pysyft library
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import numpy as np
import torch.utils.data as Data
import subprocess
logger = logging.getLogger(__name__)

#the data paths
data_train_path ="datasets/shakespeare/Leaf-preprocess/data/train/all_data_train_9.json"
data_test_path ="datasets/shakespeare/Leaf-preprocess/data/test/all_data_test_9.json"
#dataset characteristics
nb_label = 79
#define the sent140 dataset object
class Shakespeare(Dataset):
    def __init__(self, data_root,data_size = None):
        self.ALL_LETTERS = '''1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz -,\'!;"[]?().:>}&'''#符号表
        self.NUM_LETTERS = len(self.ALL_LETTERS)  
        with open(data_root) as file:
            js = json.load(file)
            self.data = []
            self.targets = []
            #self.num_samples = js['num_samples']
           
            self.user_size = len(js['num_samples'])
            self.num_samples = []
            self.data_size = data_size
            for idx,u in enumerate(js['users']):
                if idx < self.user_size:
                    self.num_samples.append(js['num_samples'][idx])
                    for d in js['user_data'][u]['x']:
                        self.data.append(self.word_to_indices(d))
                    for t in js['user_data'][u]['y']:
                        self.targets.append(self.letter_to_vec(t))
                else:
                    break
                if self.data_size != None:    
                    if len(self.data) > self.data_size:
                        break
        self.data = torch.tensor(self.data).view(-1,1,80)


    def letter_to_vec(self, letter):
        '''returns one-hot representation of given letter
        '''
        index = self.ALL_LETTERS.find(letter)
        if index == -1:
            print(letter)
        return index

    def word_to_indices(self, word):
        '''returns a list of character indices
        Args:
        word: string
        Return:
        indices: int list with length len(word)
        '''
        indices = []
        for c in word:
            n = self.ALL_LETTERS.find(c)
            indices.append(n)
            if n == -1:
                print(c)
        return indices

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        return data,target
# Load data function
def load_shakespeare(args):
    """ The load function of the dataset

    Args:
       args: to get the global dataset argument.

    Returns:
        train_loader, test_loader, global_loader: the data loaders.
    """
    train_loader = []
    test_loader = []
    global_loader = [] 
    #download data from leaf 
    subprocess.call(['datasets/shakespeare/Leaf-preprocess/preprocess.sh','-t sample'])

    if args.split_mode == "niid":
        dataset = Shakespeare(data_train_path,sum(args.data_size))
    else :
        dataset = Shakespeare(data_train_path)

    if args.global_dataset == True:
        subset_indices = random.sample(range(0, len(dataset.targets)),k=int(args.data_rate * len(dataset)))
        global_set = torch.utils.data.Subset(dataset, subset_indices)
        global_loader = torch.utils.data.DataLoader(
            global_set,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )
 
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    

    test_loader = torch.utils.data.DataLoader(
       Shakespeare(data_test_path,30000),
       batch_size=1,
       shuffle=True,
       num_workers=2
    )
        
    return train_loader, test_loader, global_loader
# Split function
def split_shakespeare(args,train_loader,global_loader=None):
    """ The split function

    Args:
       args: the arguments.
       train_loader,global_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    if args.split_mode == "iid":
        if args.add_error == True:
            temp_datasets = iid_split(args,train_loader)
            sub_datasets = addErrorDataset(args,temp_datasets)
        else:
            sub_datasets = iid_split(args,train_loader)
    else:
        if args.type == "random":
            if args.add_error == True and args.global_dataset == False:
                temp_datasets = random_split(args,train_loader)
                sub_datasets = addErrorDataset(args,temp_datasets)
            elif args.add_error == False and args.global_dataset == True:
                temp_datasets1 = random_split(args,train_loader)
                temp_datasets2 = addGlobalDataset(args,global_loader)
                sub_datasets = concat_fun(args,temp_datasets1,temp_datasets2)
            elif args.add_error == True and args.global_dataset == True:
                temp_datasets1 = random_split(args,train_loader)
                temp_datasets2 = addGlobalDataset(args,global_loader)
                temp_datasets = concat_fun(args,temp_datasets1,temp_datasets2)
                sub_datasets = addErrorDataset(args,temp_datasets)
            else:
                 sub_datasets = random_split(args,train_loader)
        elif args.type == "label":
            if args.add_error == True and args.global_dataset == False:
                temp_datasets = labels_split(args,train_loader)
                sub_datasets = addErrorDataset(args,temp_datasets)
            elif args.add_error == False and args.global_dataset == True:
                temp_datasets1 = labels_split(args,train_loader)
                temp_datasets2 = addGlobalDataset(args,global_loader)
                sub_datasets = concat_fun(args,temp_datasets1,temp_datasets2)
            elif args.add_error == True and args.global_dataset == True:
                temp_datasets1 = labels_split(args,train_loader)
                temp_datasets2 = addGlobalDataset(args,global_loader)
                temp_datasets = concat_fun(args,temp_datasets1,temp_datasets2)
                sub_datasets = addErrorDataset(args,temp_datasets)  
            else:
                sub_datasets = labels_split(args,train_loader)        
    return sub_datasets
# IID Split
def iid_split(args, train_loader):
    """ The iid split function

    Args:
       args: the arguments.
       train_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    sub_datasets = [[] for i in range(args.clients)]
    temp_datasets = [[] for i in range(nb_label)]
    node_index = 0
    #stock all the class in list
    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()

        temp_datasets[num_label].append(
            [imgs[0].numpy(), label[0].numpy()])
        if step % 5000 == 0:
            print("split dataset step: ", step)
    s = []
    for i in temp_datasets:
        s.append(len(i))
    # loop temp_datasets, add and contract

    rs = random.sample(range(0, nb_label), nb_label) # 0 - 9 random nums
    # according to client list, distribute label dataset
    all_label_kinds = len(temp_datasets)
    label_num=[]
    for i in range(args.clients):
        label_num.append(all_label_kinds)
    for index, x in enumerate(label_num):
        temp_list = []
        if x > all_label_kinds:
            x = all_label_kinds
        for y in range(x):
            # temp_list only contain 10 kinds labels
            labels_index = y  % all_label_kinds
            if args.iid_share == False:
                size = s[y]//args.clients
                temp_list.extend(temp_datasets[labels_index][:size])
                del temp_datasets[labels_index][:size]
            else:
                random.shuffle(temp_datasets[labels_index])
                size = int(len(temp_datasets[labels_index])*args.iid_rate)
                temp_list.extend(temp_datasets[labels_index][:size])
            print("Client %d" % index, "| add label-%d dataset" % (labels_index),"| dataset size %d",len(temp_list))        
        sub_datasets[index] = temp_list

    return sub_datasets
# Random split
def random_split(args, loader):
    """ The random split function

    Args:
       args: the arguments.
       loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    node_num = args.clients
    sub_datasets = [[] for i in range(node_num)]
    data_size = args.data_size
    temp_list = []
    node_index = 0
    temp_step = data_size[node_index]
    num = 0

    for step, (imgs, labels) in enumerate(loader):
        num +=1
        temp_list.append([imgs[0].numpy(), labels[0].numpy()])
        # temp_list.append([imgs.numpy(), labels.numpy()])
        if num == temp_step and num !=0:
            print("finish spliting %d dataset" % node_index)
            sub_datasets[node_index] = temp_list
            node_index = node_index + 1
            if node_index == node_num:
                break
            temp_step += data_size[node_index]
            temp_list = []
        if step == len(loader.dataset.data) -1:
            print("finish left spliting %d dataset" % node_index)
            sub_datasets[node_index] = temp_list
    return sub_datasets
# Global dataset
def addGlobalDataset(args, global_loader):
    """ The global dataset split function

    Args:
       args: the arguments.
       global_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    percent = args.data_rate
    sub_datasets = [[] for i in range(args.clients)]
    temp_list =[]

    # add other data Attention other dataset

    for i in range(args.clients):
        for step, (imgs, labels) in enumerate(global_loader):
            if step % 1000 == 0:
                print("Client %d " % i, "| step：%d, adding sub dataset" % step)
            sub_datasets[i].append([imgs[0].numpy(), labels[0].numpy()])
    print("adding globle dataset succeed!")
    return sub_datasets
# Add globle dataset
def addErrorDataset(args, array):
    """ The function to add error 

    Args:
       args: the arguments.
       array: the subset of data 
    Returns:
        array: the subset of data of each worker with error.
    """
    error_ratio = args.error_rate
    add_error_nums = [int(error_ratio * len(array[i])) for i in range(args.clients)]
    # add error data
    for i in range(args.clients):
        for index in range(add_error_nums[i]):
            if index % 100 == 0:
                print("Client %d" % i, "| step：%d, adding other error dataset" % index)
            # array        [
            #               [[imgs, label], [imgs, label]....],
            #               [[imgs, label], [imgs, label]....],
            #              ]
            real_label = array[i][index][1]
            error_label = random.choice([i for i in range(0,nb_label) if i not in [real_label]])
            array[i].append([array[i][index][0], error_label])
    print("adds some error label data succeed!")
    return array
# concatenate two datasets 
def concat_fun(args,dataset1,dataset2):
    """ The concatenate function

    Args:
       args: the arguments.
       dataset1: the first subset of data 
       dataset2: the secoud subset of data
    Returns:
        sub_datasets: the concatenate subset.
    """
    sub_datasets = [[] for i in range(args.clients)]
    for i in range(args.clients):
        sub_datasets[i] = dataset1[i]+dataset2[i]
    return sub_datasets
# Split by label
def labels_split(args, train_loader):
    """ The labels split function

    Args:
       args: the arguments.
       train_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    sub_datasets = [[] for i in range(args.clients)]
    temp_datasets = [[] for i in range(nb_label)]
    node_index = 0
    temp_class = [[] for i in range(args.clients)]
    data_size = args.data_size

    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()

        temp_datasets[num_label].append(
            [imgs[0].numpy(), label[0].numpy()])
        if step % 5000 == 0:
            print("split dataset step: ", step)

    # loop temp_datasets, add and contract
  
    rs = random.sample(range(0, nb_label), nb_label) # 0 - 9 random nums
    # according to nodes list, distribute label dataset
    all_label_kinds = len(temp_datasets)
    sum_x = 0
    for index, x in enumerate(args.label_num):
        if x > all_label_kinds:
            x = all_label_kinds
        for y in range(x):
            # temp_list only contain 10 kinds labels                
            labels_index = (y + sum_x) % all_label_kinds
            temp_class[index].append(labels_index)
        sum_x += x

    for index,i in enumerate(temp_class):
        temp_list = []
        if args.share_samples == 0 :
            for clas in i:
                temp_list.extend(temp_datasets[clas][:data_size[index]])
                print("Client %d" % index, "| add label-%d dataset" % (clas),"| size %d"%len(temp_list))

        elif args.share_samples == 1:
            for clas in i:
                random.shuffle(temp_datasets[clas])
                temp_list.extend(temp_datasets[clas][:data_size[index]])
                print("Client %d" % index, "| add label-%d dataset" % (clas),"| size %d"%len(temp_list))

        elif args.share_samples == 2:
            for clas in i:
                s = data_size[index]
                temp_list.extend(temp_datasets[clas][:s//len(i)])
                del temp_datasets[clas][:s//len(i)]
                print("Client %d" % index, "| add label-%d dataset" % (clas),"| size %d"%len(temp_list))
        sub_datasets[index] = temp_list
    return sub_datasets