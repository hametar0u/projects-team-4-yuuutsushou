from random import shuffle
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance

LOCAL = False
DATA_PATH = "cassava-leaf-disease-classification" if LOCAL else "/kaggle/input/cassava-leaf-disease-classification"

transform_list = [
    transforms.RandomRotation(100),
    transforms.ColorJitter(brightness=1, contrast=0, saturation=0, hue=0),
    transforms.RandomHorizontalFlip(p=1),
    transforms.GaussianBlur(3),
    transforms.RandomResizedCrop(size=(800,600))
]

transform = transforms.RandomChoice(transforms=transform_list)

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, eval=False):
        self.images = []
        self.start_i = 0 if eval else 2000
        self.end_i = 2000 if eval else 10000
        self.tensor_converter = transforms.ToTensor()
        with open(DATA_PATH + "/train.csv", "r") as f:
            for line in f.readlines()[self.start_i + 1:self.end_i]:
            # for line in f.readlines()[self.start_i + 1:]:
                self.images.append(line)
        
    def __getitem__(self, index):
        line = self.images[index]
        elems = line.split(',')
        file = elems[0]
        label = elems[1]
        img = Image.open(DATA_PATH + "/train_images/" + file)
        if label != 3:
            img = transform(img)
        tensor = self.tensor_converter(img)
        img.close()
        return tensor, int(label.rstrip())

    def __len__(self):
        return len(self.images)

