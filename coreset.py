from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from enum import Enum

from datasets import Dataset
from utils import id_to_idxs



class MultiTaskDataContainer:
    """
    A class that allows to iterate over a specific subset of the training data for multiple tasks. This is necessary for implementing coresets, as both the coresets and $\tilde{D}_t \cup C_{t-1} \setminus C_t$ can have data of multiple classes.
    """
    def __init__(self):
        # self.data is a dict where keys are task ids and values are tuples of (x, y) corresponding to that task
        self.data = {}

    def __getitem__(self, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[task_id]
    
    def __setitem__(self, task_id: int, data: Tuple[torch.Tensor, torch.Tensor]):
        self.data[task_id] = data

    def __iter__(self):
        """For every task in this container, yield the tuple (task_idxs, x, y) tensors."""
        for task_id in self.data:
            if len(self.data[task_id][0]):
                yield id_to_idxs(task_id), self.data[task_id][0], self.data[task_id][1]
    
    def lens(self):
        return [(task_id, len(self.data[task_id][0])) for task_id in self.data if len(self.data[task_id][0]) > 0]
    
    def __len__(self):
        return sum([len(self.data[task_id][0]) for task_id in self.data])
    
    def task_ids(self):
        return [task_id for task_id in self.data if len(self.data[task_id][0]) > 0]
        

class CoresetMethod:
    pass


class NoCoreset(CoresetMethod):
    """
    C_t is always empty and \tilde{D}_t = D_t for all t
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.device = dataset.device()
        self.dtype = dataset.dtype()
        self.task_id = -1

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.task_id >= self.dataset.num_tasks - 1:
            raise StopIteration
        self.task_id += 1

        # Observe D_t
        x_dt, y_dt = self.dataset.get_train_data(self.task_id)

        # Create \tilde{D}_t = D_t
        Dt_tilde = MultiTaskDataContainer()
        Dt_tilde[self.task_id] = (x_dt, y_dt)

        # Create C_t = \emptyset
        Ct = MultiTaskDataContainer()

        return Dt_tilde, Ct


class RandomCoreset(CoresetMethod):
    """
    To create C_t from (D_t, C_{t-1}), we randomly sample a subset of D_t of size `size` and add it to C_{t-1}, like in the paper implementation. Note that C_{t-1} is a subset of C_t.

    Both Ct and $\tilde{D_t}$ are MultiTaskDataContainer objects.
    """

    def __init__(self, dataset: Dataset, size: int):
        self.dataset = dataset
        self.size = size
        self.task_id = -1

        # initialize empty ct
        self.Ct = MultiTaskDataContainer()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        returns \tilde{D}_t, C_t
        where \tilde{D}_t = D_t \cup C_{t-1} \setminus C_t

        Both are MultiTaskDataContainer objects.
        """
        
        if self.task_id >= self.dataset.num_tasks - 1:
            raise StopIteration
        self.task_id += 1

        # Observe D_t
        x_dt, y_dt = self.dataset.get_train_data(self.task_id)

        # Permute D_t: the first `size` samples will be added to C_t and the rest will be in \tilde{D}_t
        perm = torch.randperm(x_dt.shape[0])
        x_dt = x_dt[perm]
        y_dt = y_dt[perm]

        # Create C_t from (D_t, C_{t-1})
        self.Ct[self.task_id] = (x_dt[:self.size], y_dt[:self.size])

        # Create \tilde{D}_t = D_t \cup C_{t-1} \setminus C_t = D_t \setminus C_t
        Dt_tilde = MultiTaskDataContainer()
        Dt_tilde[self.task_id] = (x_dt[self.size:], y_dt[self.size:])

        assert len(Dt_tilde[self.task_id][0]) == len(Dt_tilde[self.task_id][1]) == len(x_dt) - self.size
        assert len(self.Ct[self.task_id][0]) == len(self.Ct[self.task_id][1])

        return Dt_tilde, self.Ct
    
    def get_task_id(self):
        return self.task_id


def k_center(x: torch.Tensor, k: int):
    """
    Return the indices of the k-center points in x, using the k-center algorithm.
    Like in the paper implementation.
    https://www.sciencedirect.com/science/article/pii/0304397585902245
    """
    # Initialize centers with the first point
    centers = [0]
    # Initialize distances to infinity
    dists = torch.ones(x.shape[0], device=x.device) * float('inf')
    # Iterate over the rest of the points
    for i in range(1, k):
        # Update distances: for each point, compute the minimum distance to the current centers
        # Norm over all dimensions except the first one
        dims = tuple(range(1,len(x.shape)))
        dists = torch.min(dists, torch.norm(x - x[centers[-1]], dim=dims))
        # Update centers
        centers.append(torch.argmax(dists).item())
    return centers



class KCenterCoreset(CoresetMethod):
    """
    To create C_t from (D_t, C_{t-1}), we use the k-center algorithm to select the `size` most informative points from D_t and add them to C_{t-1}, like in the paper implementation. Note that C_{t-1} is a subset of C_t.

    Both Ct and $\tilde{D_t}$ are MultiTaskDataContainer objects.
    """

    def __init__(self, dataset: Dataset, size: int):
        self.dataset = dataset
        self.size = size    # K
        self.task_id = -1

        # initialize empty ct
        self.Ct = MultiTaskDataContainer()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        returns \tilde{D}_t, C_t
        where \tilde{D}_t = D_t \cup C_{t-1} \setminus C_t

        Both are MultiTaskDataContainer objects.
        """
        
        if self.task_id >= self.dataset.num_tasks - 1:
            raise StopIteration
        self.task_id += 1

        # Observe D_t
        x_dt, y_dt = self.dataset.get_train_data(self.task_id)

        # Get the indices of the k-center points
        centers = k_center(x_dt, self.size)
        non_centers = [i for i in range(x_dt.shape[0]) if i not in centers]

        # Create C_t from (D_t, C_{t-1})
        self.Ct[self.task_id] = (x_dt[centers], y_dt[centers])

        # Create \tilde{D}_t = D_t \cup C_{t-1} \setminus C_t = D_t \setminus C_t
        Dt_tilde = MultiTaskDataContainer()
        Dt_tilde[self.task_id] = (x_dt[non_centers], y_dt[non_centers])

        assert len(Dt_tilde[self.task_id][0]) == len(Dt_tilde[self.task_id][1]) == len(x_dt) - self.size
        assert len(self.Ct[self.task_id][0]) == len(self.Ct[self.task_id][1])

        return Dt_tilde, self.Ct
    
    def get_task_id(self):
        return self.task_id
    


class CoresetType(Enum):
    NO_CORESET = 0
    RANDOM_CORESET = 1
    KCENTER_CORESET = 2


def create_coreset_method(dataset: Dataset, coreset_type: CoresetType, size: int) -> CoresetMethod:
    if coreset_type == CoresetType.NO_CORESET:
        return NoCoreset(dataset)
    elif coreset_type == CoresetType.RANDOM_CORESET:
        return RandomCoreset(dataset, size)
    elif coreset_type == CoresetType.KCENTER_CORESET:
        return KCenterCoreset(dataset, size)
    else:
        raise Exception('Unknown coreset type!')
