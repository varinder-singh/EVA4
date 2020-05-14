import os
from torch.utils.data import Dataset
import itertools

class ImageDataSet(Dataset):

    def __init__(self, root='/home/varinder/Documents/EVA4/S14-15/Dataset/Dataset-S15/', image_loader=None, transform=None):
        self.root = root
        self.image_files = [os.listdir(os.path.join(root, 'bg{0}/fg_{1}/overlay'.format(i,j))) for i, j in itertools.product(range(1,10), range(1,101))]
        self.loader = image_loader
        self.transform = transform
       


    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return sum([len(folder) for folder in self.image_files])



    def __getitem__(self, index):
        
        if index//4000 == 0:
            bg_fldr_range = 1
            if index//40 == 0:
                fg_fldr_range = 1
                image_range = index
            elif index//40 >0 and index%40==0:
                fg_fldr_range = index //40
                image_range = 40
            elif index//40 >0 and index%40 > 0:
                print("poor choice give batch size in multiple of 40")
                return 0
        elif index//4000 >0 and index%40==0:
            bg_fldr_range = index//4000
            fg_fldr_range = 100 + index%100
            image_range = 40
        else:
            print("poor choice give batch size in multiple of 40")
            return 0

        print("Img_Range {} FG_Fldr_Range {} and BG_Fldr_Range".format(image_range, fg_fldr_range, bg_fldr_range))
        images = [self.loader(os.path.join(self.root, 'bg{}/fg_{}/overlay'.format(i+1,j+1), self.image_files[j][k])) for (i, (j, k)) in itertools.product(range(0,bg_fldr_range), itertools.product(range(0,fg_fldr_range), range(0,image_range)))]
        if self.transform is not None:
            images = [self.transform(img) for img in images]
        return images