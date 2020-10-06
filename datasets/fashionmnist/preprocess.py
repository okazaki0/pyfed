from torchvision import datasets, transforms
import torch
import random
import numpy as np
import torch.utils.data as Data

# Data storge path
DATA_PATH = "./data"
# dataset characteristics
nb_label = 10
# Load data function
def load_fashionmnist(args):
    """The load function of the dataset

    Args:
       args: to get the global dataset argument.

    Returns:
        train_loader, test_loader, global_loader: the data loaders.
    """
    train_loader = []
    test_loader = []
    global_loader = []

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = datasets.FashionMNIST(
        DATA_PATH, train=True, download=True, transform=transform_train
    )

    # select the sub set from the dataset
    if args.global_dataset == True:
        subset_indices = [
            random.randint(0, len(dataset.targets))
            for i in range(int(args.data_rate * len(dataset)))
        ]
        globle_set = torch.utils.data.Subset(dataset, subset_indices)
        global_loader = torch.utils.data.DataLoader(
            globle_set, batch_size=1, shuffle=True, num_workers=0
        )

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DATA_PATH, train=False, transform=transform_test),
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )
    return train_loader, test_loader, global_loader


# Split function
def split_fashionmnist(args, train_loader, global_loader=None):
    """The split function

    Args:
       args: the arguments.
       train_loader,global_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    if args.split_mode == "iid":
        if args.add_error == True:
            temp_datasets = iid_split(args, train_loader)
            sub_datasets = addErrorDataset(args, temp_datasets)
        else:
            sub_datasets = iid_split(args, train_loader)
    else:
        if args.type == "random":
            if args.add_error == True and args.global_dataset == False:
                temp_datasets = random_split(args, train_loader)
                sub_datasets = addErrorDataset(args, temp_datasets)
            elif args.add_error == False and args.global_dataset == True:
                temp_datasets1 = random_split(args, train_loader)
                temp_datasets2 = addGlobalDataset(args, global_loader)
                sub_datasets = concat_fun(args, temp_datasets1, temp_datasets2)
            elif args.add_error == True and args.global_dataset == True:
                temp_datasets1 = random_split(args, train_loader)
                temp_datasets2 = addGlobalDataset(args, global_loader)
                temp_datasets = concat_fun(args, temp_datasets1, temp_datasets2)
                sub_datasets = addErrorDataset(args, temp_datasets)
            else:
                sub_datasets = random_split(args, train_loader)
        elif args.type == "label":
            if args.add_error == True and args.global_dataset == False:
                temp_datasets = labels_split(args, train_loader)
                sub_datasets = addErrorDataset(args, temp_datasets)
            elif args.add_error == False and args.global_dataset == True:
                temp_datasets1 = labels_split(args, train_loader)
                temp_datasets2 = addGlobalDataset(args, global_loader)
                sub_datasets = concat_fun(args, temp_datasets1, temp_datasets2)
            elif args.add_error == True and args.global_dataset == True:
                temp_datasets1 = labels_split(args, train_loader)
                temp_datasets2 = addGlobalDataset(args, global_loader)
                temp_datasets = concat_fun(args, temp_datasets1, temp_datasets2)
                sub_datasets = addErrorDataset(args, temp_datasets)
            else:
                sub_datasets = labels_split(args, train_loader)
    return sub_datasets


# IID Split
def iid_split(args, train_loader):
    """The iid split function

    Args:
       args: the arguments.
       train_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    sub_datasets = [[] for i in range(args.clients)]
    temp_datasets = [[] for i in range(nb_label)]  # 10 = number of classes
    node_index = 0
    # stock all the class in list
    for step, (imgs, label) in enumerate(train_loader):
        num_label = label.data.item()
        temp_datasets[num_label].append([imgs[0].numpy(), label[0].numpy()])
        if step % 5000 == 0:
            print("split dataset step: ", step)
    s = []
    for i in temp_datasets:
        s.append(len(i))
    # loop temp_datasets, add and contract

    rs = random.sample(range(0, nb_label), nb_label)  # 0 - 9 random nums
    # according to client list, distribute label dataset
    all_label_kinds = len(temp_datasets)
    label_num = []
    for i in range(args.clients):
        label_num.append(all_label_kinds)
    for index, x in enumerate(label_num):
        temp_list = []
        if x > all_label_kinds:
            x = all_label_kinds
        for y in range(x):
            # temp_list only contain 10 kinds labels
            labels_index = y % all_label_kinds
            if args.iid_share == False:
                size = s[y] // args.clients
                temp_list.extend(temp_datasets[labels_index][:size])
                del temp_datasets[labels_index][:size]
            else:
                random.shuffle(temp_datasets[labels_index])
                size = int(len(temp_datasets[labels_index]) * args.iid_rate)
                temp_list.extend(temp_datasets[labels_index][:size])
            print(
                "Client %d" % index,
                "| add label-%d dataset" % (labels_index),
                "| dataset size %d",
                len(temp_list),
            )
        sub_datasets[index] = temp_list

    return sub_datasets


# Random split
def random_split(args, loader):
    """The random split function

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
        num += 1
        temp_list.append([imgs[0].numpy(), labels[0].numpy()])
        # temp_list.append([imgs.numpy(), labels.numpy()])
        if num == temp_step and num != 0:
            print("finish spliting %d dataset" % node_index)
            sub_datasets[node_index] = temp_list
            node_index = node_index + 1
            if node_index == node_num:
                break
            temp_step += data_size[node_index]
            temp_list = []
        if step == len(loader.dataset.data) - 1:
            print("finish left spliting %d dataset" % node_index)
            sub_datasets[node_index] = temp_list
    return sub_datasets


# Global dataset
def addGlobalDataset(args, global_loader):
    """The global dataset split function

    Args:
       args: the arguments.
       global_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    percent = args.data_rate
    sub_datasets = [[] for i in range(args.clients)]
    temp_list = []

    # add other data Attention other dataset

    for i in range(args.clients):
        for step, (imgs, labels) in enumerate(global_loader):
            if step % 1000 == 0:
                print("Client %d " % i, "| step：%d, adding sub dataset" % step)
            sub_datasets[i].append([imgs[0].numpy(), labels[0].numpy()])
    print("adding globle dataset succeed!")
    return sub_datasets


# Add Error dataset
def addErrorDataset(args, array):
    """The function to add error

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
            error_label = random.choice(
                [i for i in range(0, nb_label) if i not in [real_label]]
            )
            array[i].append([array[i][index][0], error_label])
    print("adds some error label data succeed!")
    return array


# concatenate two datasets
def concat_fun(args, dataset1, dataset2):
    """The concatenate function

    Args:
       args: the arguments.
       dataset1: the first subset of data
       dataset2: the secoud subset of data
    Returns:
        sub_datasets: the concatenate subset.
    """
    sub_datasets = [[] for i in range(args.clients)]
    for i in range(args.clients):
        sub_datasets[i] = dataset1[i] + dataset2[i]
    return sub_datasets


# Split by label
def labels_split(args, train_loader):
    """The labels split function

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

        temp_datasets[num_label].append([imgs[0].numpy(), label[0].numpy()])
        if step % 5000 == 0:
            print("split dataset step: ", step)

    # loop temp_datasets, add and contract
    # node_label_num [1, 2, 2, 5, 7]

    rs = random.sample(range(0, nb_label), nb_label)  # 0 - 9 random nums

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

    for index, i in enumerate(temp_class):
        temp_list = []
        if args.share_samples == 0:
            for clas in i:
                temp_list.extend(temp_datasets[clas][: data_size[index]])
                print(
                    "Client %d" % index,
                    "| add label-%d dataset" % (clas),
                    "| size %d" % len(temp_list),
                )

        elif args.share_samples == 1:
            for clas in i:
                random.shuffle(temp_datasets[clas])
                temp_list.extend(temp_datasets[clas][: data_size[index]])
                print(
                    "Client %d" % index,
                    "| add label-%d dataset" % (clas),
                    "| size %d" % len(temp_list),
                )

        elif args.share_samples == 2:
            for clas in i:
                s = data_size[index]
                temp_list.extend(temp_datasets[clas][: s // len(i)])
                del temp_datasets[clas][: s // len(i)]
                print(
                    "Client %d" % index,
                    "| add label-%d dataset" % (clas),
                    "| size %d" % len(temp_list),
                )
        sub_datasets[index] = temp_list
    return sub_datasets
