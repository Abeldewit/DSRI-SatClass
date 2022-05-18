import numpy as np
import random
import torch.utils.data as tdata
from torch.utils.data import DataLoader, random_split
import torch
import pathlib
from utilities.util_funcs import pad_tensor
from pipe import chain
from ast import literal_eval as eval
from functools import lru_cache
import os

def create_split_dataloaders(
    dataset: tdata.Dataset, 
    splits: tuple, 
    shuffle: bool = True, 
    batch_size: int = 1,
    ) -> tuple:
    
    # Create numbers for splits
    train, val = int(len(dataset)*splits[0]*splits[0]), int(len(dataset)*splits[0]*splits[1])
    test = len(dataset) - train - val
    assert train + val + test == len(dataset) # sanity check

    # Split the dataset
    train_set, val_test, test_set = random_split(
        dataset, 
        [train, val, test], 
        torch.Generator().manual_seed(42)
    )

    # Create dataloaders from datasets
    train_set, val_test, test_set = \
        DataLoader(train_set, batch_size = batch_size, shuffle = shuffle), \
        DataLoader(val_test, batch_size = batch_size, shuffle = shuffle), \
        DataLoader(test_set, batch_size = batch_size, shuffle = shuffle)
    return train_set, val_test, test_set

class PASTIS(tdata.Dataset):
    """
    Data loader for PASTIS dataset. With customization options for the data output. 
    Args:
        path_to_pastis: path to the folder containing the data
        data_files: path to the file containing the data
        label_files: path to the file containing the labels
        pad: whether to pad the tensor to the max_t
        rgb_only: whether to only use the RGB channel
        multi_temporal: whether to use multi-temporal data

    """
    def __init__(
        self, 
        path_to_pastis:str, 
        data_files: str, 
        label_files: str, pad: bool=False, 
        rgb_only: bool=False, 
        multi_temporal: bool = True,
        cache: str = './utilities/'
    ) -> None:
        super(PASTIS, self).__init__()
        # Path and folder names
        self.folder = path_to_pastis
        self.data_files = data_files
        self.label_files = label_files
        self.rgb = rgb_only
        self.multi_temporal = multi_temporal
        self.cache_dir = cache

        # File structure with path and file names
        self.__file_structure = {
            'DATA_S2': (pathlib.Path(self.folder, 'DATA_S2'), 'S2_*.npy'),
            'ANNOTATIONS': (pathlib.Path(self.folder, 'ANNOTATIONS'), 'TARGET_*.npy'),
            'HEATMAP': (pathlib.Path(self.folder, 'INSTANCE_ANNOTATIONS'), 'HEATMAP_*.npy'),
            'INSTANCES': (pathlib.Path(self.folder, 'INSTANCE_ANNOTATIONS'), 'INSTANCES*.npy'),
            'ZONES': (pathlib.Path(self.folder, 'INSTANCE_ANNOTATIONS'), 'ZONES_*.npy'),
        }
        self.combination = self.create_combination()

        # Parameters 
        self.max_t = 61 # max time steps for padding
        self.pad = pad # whether to pad or not

        
    def __len__(self) -> int:
        return len(self.combination)   
        
    def __iter__(self)-> iter:
        self.counter = 0
        return self

    def __next__(self) -> tuple:
        if self.counter >= len(self):
            raise StopIteration
        else:
            if not self.multi_temporal:
                x_path, y_path, time = self.combination[self.counter]
            else:
                x_path, y_path = self.combination[self.counter]
            x, y = np.load(x_path), np.load(y_path)
            x,y = torch.from_numpy(x.astype(np.float32)), \
                    torch.from_numpy(y.astype(np.float32))

            if y.shape[0] == 3:
                y = y[0, :, :] 

            if self.rgb:
                x = x[:, [2, 1, 0], :, :]

            if not self.multi_temporal:
                x = x[time, :]
            
            self.counter += 1
            return (pad_tensor(x, self.max_t, pad_value=0), y) if self.pad else (x, y)
    
    def __getitem__(self, item):
        if not self.multi_temporal:
            x_path, y_path, time = self.combination[item]
        else:
            x_path, y_path = self.combination[item]
        x, y = np.load(x_path), np.load(y_path)
        x, y = torch.from_numpy(x.astype(np.float32)), \
                torch.from_numpy(y.astype(np.float32))
        
        if y.shape[0] == 3:
            y = y[0, :, :]

        if self.rgb:
            x = x[:, [2, 1, 0], :, :]
        
        if not self.multi_temporal:
            x = x[time, :]
        
        return (pad_tensor(x, self.max_t, pad_value=0), y) if self.pad else (x, y)

    @lru_cache(maxsize=None)
    def create_combination(self) -> list:
        # Only check is whether to use time or not
        if self.multi_temporal:
            _file = os.path.join(self.cache_dir, 'time.txt')
        else:
            _file = os.path.join(self.cache_dir, 'no_time.txt')
        
        with open(_file, 'r') as f:
            time_combined = [eval(line) for line in f.readlines()]
            return time_combined

    # def create_combination_old(self):
    #     ## OBSCURE FUNCTION, DO NOT USE
    #     ## PATHS AND FILE NAMES ARE HARDCODED

    #     # Get the path to the wanted files
    #     data_structure = self.__file_structure[self.data_files]
    #     label_structure = self.__file_structure[self.label_files]

    #     # Get the file names
    #     x_values = list(data_structure[0].glob(data_structure[1]))
    #     y_values = list(label_structure[0].glob(label_structure[1]))

    #     # Get the file name ID's 
    #     x_ids = {x.name.split('_')[-1].split('.')[0]: x for x in x_values}
    #     y_ids = {y.name.split('_')[-1].split('.')[0]: y for y in y_values}

    #     # Assert if there is the same amount of files
    #     assert len(set(x_ids.keys()) - set(y_ids.keys())) == len(x_values) - len(y_values)

    #     # Create a set of the ID's that both folders share
    #     common_ids = list(set(x_ids.keys()) - set(set(x_ids.keys()) - set(y_ids.keys())))
    #     # Use the common ids to create a combination of the files
    #     combined = [(x_ids[idx], y_ids[idx]) for idx in common_ids]
        
    #     # If not using time steps, create single samples of each time step (takes a while)
    #     if self.no_time:
    #         combined = list([[(combi[0], combi[1], i) for i in range(np.load(combi[0]).shape[0])] for combi in combined] | chain)
    #     return combined


if __name__ == "__main__":
    dl = PASTIS(
        path_to_pastis='/Users/abel/Coding/Capgemini/DSRI-SatClass/data/PASTIS',
        rgb_only=True, 
        multi_temporal=True
        )