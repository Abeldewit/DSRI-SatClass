from attr import has
import numpy as np
import torch.utils.data as tdata
from torch.utils.data import DataLoader
import torch
import os
import json
from datetime import datetime
import multiprocessing as mp
import torchvision
import torchvision.transforms.functional as TF
import random

import matplotlib.pyplot as plt

class PASTIS(tdata.Dataset):
    def __init__(
        self, 
        path_to_pastis:str, 
        data_files: str, 
        label_files: str,
        rgb_only: bool=False, 
        multi_temporal: bool = True,
        fold: int = 1,
        reference_date="2018-09-01",
        subset_type = 'train',
        device: str = 'cpu',
        shuffle: bool = False,
        pre_load = False,
        remove_clouds = False,
        norm = True, # Normalize the data
        augment = False, # Augment the data
    ) -> None:
        """
        Data loader for PASTIS dataset. With customization options for the data output. 
        Args:
            path_to_pastis (str): path to the folder containing the data
            data_files (str): path to the file containing the data
            label_files (str): path to the file containing the labels
            rgb_only (bool): whether to only use the RGB channel, true will yield (3, H, W), false will yield (10, H, W)
            multi_temporal (bool): whether to use multi-temporal data, true will yield a shape of (t, c, h, w), false will yield a shape of (c, h, w)
            cache_dir (str): path to the cache directory, default is './utilities/'
            fold (int): which fold to use, default is 1
            subset_type (str): which subset to use, default is 'train', other options are 'test', 'val'
            device (str): which device to load the tensors to, default is 'cpu'
        """
        super(PASTIS, self).__init__()
        # Path and folder names
        self.folder = path_to_pastis
        self.data_files = data_files
        self.label_files = label_files
        self.rgb = rgb_only
        self.multi_temporal = multi_temporal
        self.remove_clouds = remove_clouds
        self.fold = fold
        self.reference_date = datetime.strptime(reference_date, '%Y-%m-%d')
        self.subset_type = subset_type
        self.device = device
        self.pre_load = pre_load
        self.done_loading = False if self.pre_load else True
        self.augment = augment

        # Parameters 
        self.max_t = 61 # max time steps for padding
        self.pad = multi_temporal # whether to pad or not (depends on multi_temporal)

        if not self.multi_temporal and self.pad:
            raise ValueError('Padding is only neccesary for multi-temporal data.')
        
        if self.remove_clouds:
            self.no_clouds = json.load(open(os.path.join(self.folder, 'cloud_dict.json')))

        self.metadata = self._read_metadata()
        self.combination = self._create_combination()

        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.combination)

        if self.pre_load:
            self.load_all()
            self.done_loading = True

        if norm:
            with open(
                os.path.join(self.folder, "NORM_S2_patch.json"), "r"
            ) as file:
                normvals = json.loads(file.read())[f'Fold_{self.fold}']
            
            means = torch.tensor(normvals['mean'])
            stds = torch.tensor(normvals['std'])
            if self.rgb:
                means = means[[2, 1, 0]]
                stds = stds[[2, 1, 0]]

            self.norm = torchvision.transforms.Normalize(means, stds)

    def load_all(self):
        """
        Loads all the data into memory.
        """
        max_cpu = min(50, mp.cpu_count())
        with mp.Pool(mp.cpu_count()) as pool:
            combi_as_args = [(c,) for c in self.combination]
            completed = pool.starmap(self._read_files, combi_as_args)

        self.loaded_data = completed

    def __len__(self) -> int:
        return len(self.combination)   
        
    def __iter__(self)-> iter:
        self.counter = -1
        if self.shuffle:
            np.random.shuffle(self.combination)
        return self

    def __next__(self) -> tuple:
        self.counter += 1
        if self.counter >= len(self):
            raise StopIteration
        else:
            return self.__getitem__(self.counter)
    
    def __getitem__(self, item):
        if not isinstance(item, int):
            raise TypeError('Item must be an integer.')
        if item >= len(self):
            raise IndexError('Item out of range.')

        # If we're pre-loading, skipp all below and retrieve from ram
        if self.pre_load and self.done_loading:
            data =  self.loaded_data[item]
            x, y, time = data
            # x.to(self.device)
            # y.to(self.device)
            return x, y, time
        
        else:
            return self._read_files(self.combination[item])

        
    def _read_files(self, sample):
        x_path, y_path, time = sample['data_path'], sample['label_path'], sample['time']
        
        if self.multi_temporal:
            time = [datetime.strptime(str(d), '%Y%m%d') for d in time.values()]
            diff = [(d - self.reference_date).days for d in time]
            time = diff

        # Load the data from the file and convert to tensor
        x, y = np.load(x_path), np.load(y_path)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        # The labels may have more data than just the classes, 
        # so we need to get the correct shape.
        if y.shape[0] == 3: y = y[0, :, :]
        
        # If we are using only RGB channels, 
        # we need to select and swap the channels.
        if self.rgb: x = x[:, [2, 1, 0], :, :]

        if self.multi_temporal:
            padlen = self.max_t - x.shape[0]
            x = np.pad(x, ((0, padlen), (0, 0), (0, 0), (0, 0)), 'constant')
            x = torch.from_numpy(x.astype(np.float32))
            time = np.pad(time, (0, self.max_t - len(time)), 'constant').flatten()
            time = torch.from_numpy(time.astype(np.float32))

        elif not self.multi_temporal:
            x = x[time, :, :, :]

        if hasattr(self, 'norm'):
            x = self.norm(x)

        if self.augment:
            x, y = self._augment(x, y)
        
        return x, y, time

    def _augment(self, image, mask):

        # The images that are multi-tempral have to be transformed
        # on a time to time basis
        def _transform_temporal(image, transform, **kwargs):
            new_im = []
            for i in range(image.shape[0]):
                _t = transform(image[i], **kwargs)
                new_im.append(_t)
            return torch.stack(new_im, axis=0)
        
        # Randomly flip the image
        if random.random() > 0.5:
            if self.multi_temporal:
                image = _transform_temporal(image, TF.hflip)
            else: 
                image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Randomly flip the image vertically
        if random.random() > 0.5:
            if self.multi_temporal:
                image = _transform_temporal(image, TF.vflip)
            else:
                image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Randomly rotate the image
        if random.random() > 0.5:
            angle = random.randint(-180, 180)
            if self.multi_temporal:
                image = _transform_temporal(image, TF.rotate, angle=angle, fill=0)
            else:
                image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask.view((1, *mask.shape)), angle, fill=0)[0]

        return image, mask


    def _create_combination(self):
        # Create the feature sets based on the fold number
        train, test, val = self._create_folds()

        # The dataset will only be one type of subset
        features = None
        if self.subset_type == 'train':
            features = train
        elif self.subset_type == 'test':
            features = test
        elif self.subset_type == 'val':
            features = val
        else:
            raise ValueError('subset_type must be one of train, test, val')

        # Create the combination of the features and the time steps
        # with all time steps combined in one sample (multi-temporal).
        multi_temporal_combinations = [
            {
                'data_path': os.path.join(self.folder, self.data_files, 'S2_{}.npy'.format(f['properties']['ID_PATCH'])),
                'label_path': os.path.join(self.folder, self.label_files, 'TARGET_{}.npy'.format(f['properties']['ID_PATCH'])),
                'time': f['properties']['dates-S2'],
            }
            for f in features
        ]

        # Create the combination of the features and the time steps,
        # with a sample for each time step.
        no_temporal_combinations = []
        for cmb in multi_temporal_combinations:
            time_length = len(cmb['time'])
            for i in range(time_length):
                f_name = cmb['data_path'].split('/')[-1].replace('.npy', '')
                if hasattr(self, 'no_clouds'):
                    if f_name in self.no_clouds:
                        if i in self.no_clouds[f_name]:
                            continue
                no_temporal_combinations.append(
                    {
                        'data_path': cmb['data_path'],
                        'label_path': cmb['label_path'],
                        'time': i,
                    }
                )
        
        # Return either, based on the multi_temporal flag.
        if self.multi_temporal:
            return multi_temporal_combinations
        else:
            return no_temporal_combinations

    def _read_metadata(self) -> dict:
        """
        This function will read the metadata file and set the class variables.
        """
        with open(os.path.join(self.folder, 'metadata.geojson'), 'r') as f:
            metadata = json.load(f)
            return metadata

    def _create_folds(self, folds: dict = None) -> None:
        """
        This function will create the folds for the cross validation.
        The folds are predefined in 'metadata.geojson', but can be overwritten
        Args:
            folds: A dictionary with the folds.
        """
        if not folds:
            folds = {
                1: {'train': (1, 2, 3), 'test': 4, 'val': 5},
                2: {'train': (2, 3, 4), 'test': 5, 'val': 1},
                3: {'train': (3, 4, 5), 'test': 1, 'val': 2},
                4: {'train': (4, 5, 1), 'test': 2, 'val': 3},
                5: {'train': (5, 1, 2), 'test': 3, 'val': 4},
            }

        # Get the fold id for each instance
        fold_indexes = np.array([f['properties']['Fold'] for f in self.metadata['features']])

        # Get the indices for each fold
        train_indexes = np.where(np.isin(fold_indexes, [folds[self.fold]['train']]))[0]
        test_indexes = np.where(fold_indexes == folds[self.fold]['test'])[0]
        val_indexes = np.where(fold_indexes == folds[self.fold]['val'])[0]

        # Get the feature data for each fold
        train_features = np.take(self.metadata['features'], train_indexes)
        test_features = np.take(self.metadata['features'], test_indexes)
        val_features = np.take(self.metadata['features'], val_indexes)

        return train_features, test_features, val_features


        
        


if __name__ == "__main__":
    dl = PASTIS(
        path_to_pastis='/Users/abel/Coding/Capgemini/DSRI-SatClass/src/data/PASTIS',
        data_files='DATA_S2',
        label_files='ANNOTATIONS',
        rgb_only=True, 
        multi_temporal=True,
        augment=True,
        )
    next(iter(dl))