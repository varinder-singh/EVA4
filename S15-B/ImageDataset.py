import os
from torch.utils.data import Dataset
import itertools
import csv

class ImageDataSet(Dataset):

    def __init__(self, root='./', image_loader=None, transform=None):
        self.root = root+'label_data.csv'
        print("Hooo Laaa")
        self.dataset = []
        with open(self.root) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for col in csv_reader:
                self.dataset.append(col)
                line_count += 1
        self.loader = image_loader
        self.transform = transform


    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, index):
        batch_data = self.dataset[:index]
        batch_images = [self.loader(batch_data[i][j]) for i, j in itertools.product(range(len(batch_data)), range(4))]
        if self.transform is not None:
            batch_images = [self.transform(img) for img in batch_images]
        return batch_images