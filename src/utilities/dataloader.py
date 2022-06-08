import numpy as np
# from .util_funcs import pad_tensor
import os
import json
from datetime import datetime
import multiprocessing as mp
import torch
from torch.nn import functional as F

def create_split_dataloaders(*args, **kwargs):
    """
    Create dataloaders for training and validation.
    """
    fold = kwargs.get('fold', 1)
    shuffle = kwargs.get('shuffle', False)
    batch_size = kwargs.get('batch_size', 1)

    train_set = FastDataLoader(*args, **kwargs, subset_type='train')
    val_set = FastDataLoader(*args, **kwargs, subset_type='val')
    test_set = FastDataLoader(*args, **kwargs, subset_type='test')

    return train_set, val_set, test_set


class FastDataLoader:
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
        shuffle: bool = True,
        batch_size: int = 1,
        ):
        # Path names
        self.folder = path_to_pastis
        self.data_files = data_files
        self.label_files = label_files

        # Data type parameters
        self.rgb = rgb_only
        self.multi_temporal = multi_temporal

        # Data split parameters
        self.fold = fold
        self.subset_type = subset_type

        # Temporal parameters
        self.max_t = 61 # The maximum number of time steps
        self.reference_date = datetime.strptime(reference_date, '%Y-%m-%d')

        # Batch parameters
        self.batch_size = batch_size

        # Data loading functions
        self.metadata = self._read_metadata()
        self.combination = self._create_combination()
        if shuffle:
            np.random.shuffle(self.combination)

    def __len__(self) -> int:
        return len(self.combination)

    def __iter__(self):
        self.counter = -1
        return self

    def __next__(self):
        if self.counter >= len(self):
            raise StopIteration
        self.counter += 1
        
        # Get the combination for the current counter
        bracket = (self.counter*self.batch_size, (self.counter+1)*self.batch_size)
        batch = [(b,) for b in self.combination[bracket[0]:bracket[1]]]
        
        # Multi-process reading the data
        cpu_count = min(self.batch_size, mp.cpu_count())
        with mp.Pool(cpu_count) as pool:
            completed = pool.starmap(self._read_files, batch)
        
        # Stack and convert to tensor
        x = np.stack([c[0] for c in completed]).astype(np.float32)
        x = torch.from_numpy(x)
        y = np.stack([c[1] for c in completed]).astype(np.float32)
        y = torch.from_numpy(y)
        time = np.stack([c[2] for c in completed])

        return x, y, time
        
    def _read_files(self, sample):
        x_path, y_path, time = sample['data_path'], sample['label_path'], sample['time']
        
        if self.multi_temporal:
            time = [datetime.strptime(str(d), '%Y%m%d') for d in time.values()]
            diff = [(d - self.reference_date).days for d in time]
            time = diff

        # Load the data from the file and convert to tensor
        x, y = np.load(x_path), np.load(y_path)
        # x = torch.from_numpy(x.astype(np.float32))
        # y = torch.from_numpy(y.astype(np.float32))

        # The labels may have more data than just the classes, 
        # so we need to get the correct shape.
        if y.shape[0] == 3: y = y[0, :, :]
        
        # If we are using only RGB channels, 
        # we need to select and swap the channels.
        if self.rgb: x = x[:, [2, 1, 0], :, :]

        if self.multi_temporal:
            padlen = self.max_t - x.shape[0]
            x = np.pad(x, ((0, padlen), (0, 0), (0, 0), (0, 0)), 'constant')
            time = np.pad(time, (0, self.max_t - len(time)), 'constant').flatten()

        elif not self.multi_temporal:
            x = x[time, :, :, :]
        
        return x, y, time


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
    dl = FastDataLoader(
        path_to_pastis='/Users/abel/Coding/Capgemini/DSRI-SatClass/src/data/PASTIS',
        data_files='DATA_S2',
        label_files='ANNOTATIONS',
        rgb_only=True, 
        multi_temporal=True,
        shuffle=True,
        batch_size=32,
    )

    for i, data in enumerate(dl):
        x, y, t = data
        print(i, x.shape, y.shape, len(t))
        if i > 3:
            break