import torch 
from torch.utils import data
import numpy as np

class DataSetBuilder(data.Dataset):
    """TinyImagenet dataset."""

    def __init__(self, rootpath, train=True, transform=None):
        """
        Args:
            rootpath: Path to the pytorch file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = rootpath
        self.transform = transform
        self.train = train
        # Load input data
        if self.train:
            self.X_ = np.load(self.path +'x_train.npy')
        else:
            self.X_ = np.load(self.path +'x_test.npy')
        # Load target data
        if self.train:
            self.y_ = np.load(self.path +'y_train.npy')
        else:
            self.y_ = np.load(self.path +'y_test.npy')

    def __len__(self):
        if self.train:
            dataFile = self.path + 'x_train.npy'
        else:
            dataFile = self.path + 'x_test.npy'
            
        data = np.load(dataFile)
        return data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.X_[idx, :, :, :]
        y = self.y_[idx]
        if self.transform is not None:
            X = self.transform(X)
        return X, torch.from_numpy(y).type(torch.LongTensor)