import numpy as np
import random
import torch.utils.data as tdata
from torch.utils.data import DataLoader, random_split
import torch
import pathlib
from utilities.util_funcs import pad_tensor

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
    def __init__(self, path_to_pastis:str, data_files: str, label_files: str, pad: bool=False, rgb_only: bool=False, no_time: bool = True) -> None:
        # Path and folder names
        self.folder = path_to_pastis
        self.data_files = data_files
        self.label_files = label_files
        self.rgb = rgb_only
        self.no_time = no_time

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
            x_path, y_path = self.combination[self.counter]
            x, y = np.load(x_path), np.load(y_path)
            x,y = torch.from_numpy(x.astype(np.float32)), \
                    torch.from_numpy(y.astype(np.float32))

            if y.shape[0] == 3:
                y = y[0, :, :] 

            if self.rgb:
                x = x[:, [2, 1, 0], :, :]

            if self.no_time:
                x = x[0, :]
            
            return (pad_tensor(x, self.max_t, pad_value=0), y) if self.pad else (x, y)
    
    def __getitem__(self, item):
        x_path, y_path = self.combination[item]
        x, y = np.load(x_path), np.load(y_path)
        x, y = torch.from_numpy(x.astype(np.float32)), \
                torch.from_numpy(y.astype(np.float32))
        
        if y.shape[0] == 3:
            y = y[0, :, :]

        if self.rgb:
            x = x[:, [2, 1, 0], :, :]
        
        if self.no_time:
            x = x[0, :]
        
        return (pad_tensor(x, self.max_t, pad_value=0), y) if self.pad else (x, y)


    def create_combination(self):
        data_structure = self.__file_structure[self.data_files]
        label_structure = self.__file_structure[self.label_files]

        x_values = list(data_structure[0].glob(data_structure[1]))
        y_values = list(label_structure[0].glob(label_structure[1]))

        x_ids = {x.name.split('_')[-1].split('.')[0]: x for x in x_values}
        y_ids = {y.name.split('_')[-1].split('.')[0]: y for y in y_values}

        assert len(set(x_ids.keys()) - set(y_ids.keys())) == len(x_values) - len(y_values)
        common_ids = list(set(x_ids.keys()) - set(set(x_ids.keys()) - set(y_ids.keys())))
        combined = [(x_ids[idx], y_ids[idx]) for idx in common_ids]
        return combined