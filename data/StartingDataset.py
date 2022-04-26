from random import shuffle
import torch
from torchvision import transforms
from PIL import Image


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, eval=False):
        self.images = []
        self.start_i = 0 if eval else 20
        self.end_i = 20 if eval else 40
        self.tensor_converter = transforms.ToTensor()
        with open("cassava-leaf-disease-classification/train.csv", "r") as f:
            for line in f.readlines()[self.start_i + 1:self.end_i]:
                self.images.append(line)
        

    def __getitem__(self, index):
        line = self.images[index]
        elems = line.split(',')
        file = elems[0]
        label = elems[1]
        img = Image.open("cassava-leaf-disease-classification/train_images/" + file)
        tensor = self.tensor_converter(img)
        return tensor, int(label.rstrip())

    def __len__(self):
        return len(self.images)

