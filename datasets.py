import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import gzip
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from enum import Enum

torch.manual_seed(0)


class Dataset():
    def to(self, device):
        raise NotImplementedError
    
    def get_train_data(self, task_id):
        """Return train data and task_idxs for the given task id"""
        raise NotImplementedError
    
    def get_test_data(self, task_id):
        """Return test data and task_idxs for the given task id"""
        raise NotImplementedError
    
    def get_non_batch_dims(self):
        raise NotImplementedError
    
    def device(self):
        raise NotImplementedError
    
    def dtype(self):
        raise NotImplementedError


class SplitMnist(Dataset):
    def __init__(self):
        x_transform = transforms.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,))
        ])

        train_set = torchvision.datasets.MNIST(root='./mnist-data', train=True, download=True)
        test_set = torchvision.datasets.MNIST(root='./mnist-data', train=False, download=True)

        self.X_train = x_transform(train_set.train_data)  # (60000, 28, 28)
        self.Y_train = train_set.train_labels             # (60000,)
        self.X_test = x_transform(test_set.test_data)     # (10000, 28, 28)
        self.Y_test = test_set.test_labels                # (10000,)

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.num_tasks = len(self.sets_0)

    def to(self, device):
        self.X_train = self.X_train.to(device)
        self.Y_train = self.Y_train.to(device)
        self.X_test = self.X_test.to(device)
        self.Y_test = self.Y_test.to(device)
        return self
    
    def get_non_batch_dims(self):
        return self.X_train.shape[1:]
    
    def device(self):
        return self.X_train.device

    def dtype(self):
        return self.X_train.dtype

    def get_train_data(self, task_id):
        """Return train data for the given task id"""
        if not 0 <= task_id < self.num_tasks:
            raise Exception('Task ID out of bounds!')

        train_0_id = self.Y_train == self.sets_0[task_id]
        train_1_id = self.Y_train == self.sets_1[task_id]

        x_train = torch.cat((
            self.X_train[train_0_id],
            self.X_train[train_1_id]
        ))
        y_train = torch.cat((
            torch.zeros((torch.count_nonzero(train_0_id),), dtype=torch.int64, device=x_train.device),
            torch.ones((torch.count_nonzero(train_1_id),), dtype=torch.int64, device=x_train.device),
        ))
        # make y_train a (B, 2) tensor with a one-hot encoding
        y_train = F.one_hot(y_train, num_classes=2).to(torch.float32)

        return x_train, y_train
    
    def get_test_data(self, task_id):
        """Return test data for the given task id"""
        if not 0 <= task_id < self.num_tasks:
            raise Exception('Task ID out of bounds!')

        test_0_id = self.Y_test == self.sets_0[task_id]
        test_1_id = self.Y_test == self.sets_1[task_id]

        x_test = torch.cat((
            self.X_test[test_0_id],
            self.X_test[test_1_id]
        ))
        y_test = torch.cat((
            torch.zeros((torch.count_nonzero(test_0_id),), dtype=torch.int64, device=x_test.device),
            torch.ones((torch.count_nonzero(test_1_id),), dtype=torch.int64, device=x_test.device),
        ))
        # make y_test a (B, 2) tensor with a one-hot encoding
        y_test = F.one_hot(y_test, num_classes=2).to(torch.float32)

        return x_test, y_test
    
    def get_task_idxs(self, task_id):
        if not 0 <= task_id < self.num_tasks:
            raise Exception('Task ID out of bounds!')

        return (self.sets_0[task_id], self.sets_1[task_id])
    
    def get_data(self, task_id):
        """Return train data, test data and task idxs for the given task id"""
        x_train, y_train = self.get_train_data(task_id)
        x_test, y_test = self.get_test_data(task_id)
        task_idxs = self.get_task_idxs(task_id)
        return x_train, y_train, x_test, y_test, task_idxs
    

class SplitFashionMnist(Dataset):
    def __init__(self):
        x_transform = transforms.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,))
        ])

        train_set = torchvision.datasets.FashionMNIST(root='./mnist-data', train=True, download=True)
        test_set = torchvision.datasets.FashionMNIST(root='./mnist-data', train=False, download=True)

        self.X_train = x_transform(train_set.train_data)  # (60000, 28, 28)
        self.Y_train = train_set.train_labels             # (60000,)
        self.X_test = x_transform(test_set.test_data)     # (10000, 28, 28)
        self.Y_test = test_set.test_labels                # (10000,)

        print(self.X_train.shape, self.Y_train.shape, self.X_test.shape, self.Y_test.shape)

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.num_tasks = len(self.sets_0)

    def to(self, device):
        self.X_train = self.X_train.to(device)
        self.Y_train = self.Y_train.to(device)
        self.X_test = self.X_test.to(device)
        self.Y_test = self.Y_test.to(device)
        return self
    
    def get_non_batch_dims(self):
        return self.X_train.shape[1:]
    
    def device(self):
        return self.X_train.device

    def dtype(self):
        return self.X_train.dtype

    def get_train_data(self, task_id):
        """Return train data for the given task id"""
        if not 0 <= task_id < self.num_tasks:
            raise Exception('Task ID out of bounds!')

        train_0_id = self.Y_train == self.sets_0[task_id]
        train_1_id = self.Y_train == self.sets_1[task_id]

        x_train = torch.cat((
            self.X_train[train_0_id],
            self.X_train[train_1_id]
        ))
        y_train = torch.cat((
            torch.zeros((torch.count_nonzero(train_0_id),), dtype=torch.int64, device=x_train.device),
            torch.ones((torch.count_nonzero(train_1_id),), dtype=torch.int64, device=x_train.device),
        ))
        # make y_train a (B, 2) tensor with a one-hot encoding
        y_train = F.one_hot(y_train, num_classes=2).to(torch.float32)

        return x_train, y_train
    
    def get_test_data(self, task_id):
        """Return test data for the given task id"""
        if not 0 <= task_id < self.num_tasks:
            raise Exception('Task ID out of bounds!')

        test_0_id = self.Y_test == self.sets_0[task_id]
        test_1_id = self.Y_test == self.sets_1[task_id]

        x_test = torch.cat((
            self.X_test[test_0_id],
            self.X_test[test_1_id]
        ))
        y_test = torch.cat((
            torch.zeros((torch.count_nonzero(test_0_id),), dtype=torch.int64, device=x_test.device),
            torch.ones((torch.count_nonzero(test_1_id),), dtype=torch.int64, device=x_test.device),
        ))
        # make y_test a (B, 2) tensor with a one-hot encoding
        y_test = F.one_hot(y_test, num_classes=2).to(torch.float32)

        return x_test, y_test
    
    def get_task_idxs(self, task_id):
        if not 0 <= task_id < self.num_tasks:
            raise Exception('Task ID out of bounds!')

        return (self.sets_0[task_id], self.sets_1[task_id])
    
    def get_data(self, task_id):
        """Return train data, test data and task idxs for the given task id"""
        x_train, y_train = self.get_train_data(task_id)
        x_test, y_test = self.get_test_data(task_id)
        task_idxs = self.get_task_idxs(task_id)
        return x_train, y_train, x_test, y_test, task_idxs
    

class DatasetType(Enum):
    SPLIT_MNIST = 1
    SPLIT_FASHION_MNIST = 2

def create_dataset(dataset_type: DatasetType):
    if dataset_type == DatasetType.SPLIT_MNIST:
        return SplitMnist()
    elif dataset_type == DatasetType.SPLIT_FASHION_MNIST:
        return SplitFashionMnist()
    else:
        raise Exception('Invalid dataset type')