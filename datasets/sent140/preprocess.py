import warnings
warnings.filterwarnings('ignore') 

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
import pandas as pd
import sys
import random
import subprocess

sys.path.append('./')

#the data paths
VOCAB_ROOT = 'datasets/sent140/Leaf-preprocess/embs.json'
DATA_ROOT_TRAIN = "datasets/sent140/Leaf-preprocess/data/all_data/training.json"
DATA_ROOT_TEST = "datasets/sent140/Leaf-preprocess/data/all_data/test.json"

#dataset characteristics
nb_label = 2
#get the vocabulary
def get_vocab(path):
    with open(path, 'r') as inf:
        embs = json.load(inf)
    vocab = embs['vocab']
    word_emb_arr = embs['emba']
    indd = {}
    for i in range(len(vocab)):
        indd[vocab[i]] = i
    vocab = {w: i for i, w in enumerate(embs['vocab'])}
    return word_emb_arr, indd, vocab
#define the sent140 dataset object
class sentiment140(Dataset):
    def __init__(self, data_root,word_emb_arr,indd,vocab,data_size=None):
        self.word_emb_arr,self.indd,self.vocab = word_emb_arr,indd,vocab
        print(data_root)
        with open(data_root) as file:
            js = json.load(file)
            self.data = []
            self.targets = []
            self.length = []
            #self.num_samples = js['num_samples']
            self.user_size = len(js['num_samples'])
            self.num_samples = []
            self.data_size = data_size
            for idx, u in enumerate(js['users']):
                if idx < self.user_size:
                    self.num_samples.append(js['num_samples'][idx])
                    for d in js['user_data'][u]['x']:
                        emba, length = self.line_to_indices(d[4])
                        self.data.append(emba)
                        self.length.append(length)
                    self.targets += js['user_data'][u]['y']
                else:
                    break
                if data_size != None :
                    if len(self.data) > self.data_size:
                        break


        self.data = torch.tensor(self.data)
        #self.targets = list(zip(self.targets,self.length))


    def split_line(self,line):
        return re.findall(r"[\w']+|[.,!?;]", line)


    def line_to_indices(self,line,  max_words=25):
        unk_id = len(self.indd)
        line_list = self.split_line(line)  # split phrase in words
        indl = [self.indd[w] if w in self.indd else unk_id for w in line_list[:max_words]]
        length = len(indl)
        indl += [unk_id] * (max_words - len(indl))
        emba = []
        for i in range(0,len(indl)):
            emba.append(self.word_emb_arr[indl[i]])
        return emba,length


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        target = self.targets[idx]
        return data,target
# Load data function
def load_sent140(args):
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
    subprocess.call(['datasets/sent140/Leaf-preprocess/preprocess.sh'])
    word_emb_arr,indd,vocab = get_vocab(VOCAB_ROOT)
    if args.split_mode == "niid":
        dataset = sentiment140(DATA_ROOT_TRAIN,word_emb_arr,indd,vocab,sum(args.data_size))
    else :
        dataset = sentiment140(DATA_ROOT_TRAIN,word_emb_arr,indd,vocab,data_size=100000)
    #select the sub set from the dataset
    if args.global_dataset == True:
        subset_indices = [random.randint(0, len(dataset.targets))for i in range(int(args.data_rate * len(dataset)))]
        globle_set = torch.utils.data.Subset(dataset, subset_indices)
        global_loader = torch.utils.data.DataLoader(
            globle_set,
            batch_size=1,
            shuffle=True,
            num_workers=0
        )
   
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        sentiment140(DATA_ROOT_TEST,word_emb_arr,indd,vocab),
        batch_size=1,
        shuffle=True,
    )
    return train_loader, test_loader, global_loader
# Split function
def split_sent140(args,train_loader,global_loader=None):
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
        if args.share_samples == 0 or args.share_samples == 3:
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