from random import shuffle, randint
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance

LOCAL = True
DATA_PATH = "cassava-leaf-disease-classification" if LOCAL else "/kaggle/input/cassava-leaf-disease-classification"

transform_list = [
    # transforms.RandomRotation(100),
    transforms.ColorJitter(brightness=1, contrast=0, saturation=0, hue=0),
    transforms.ColorJitter(brightness=0, contrast=1, saturation=0, hue=0),
    transforms.RandomHorizontalFlip(p=1),
    transforms.GaussianBlur(3),
    transforms.RandomResizedCrop(size=(600,800))
]

transform = transforms.RandomChoice(transforms=transform_list)

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 21.4k 3x224x224 black images (all zeros).
    """

    def __init__(self, eval=False):
        self.images = []
        self.start_i = 0 if eval else 2000
        self.end_i = 2000 if eval else 100000
        self.tensor_converter = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        img = img.resize((448, 448))
        img = self.tensor_converter(img)
        img = torch.reshape(img, (3, 448, 448))
        if not eval:
            num = randint(1,6)
            if num != 3:
                img = transform(img)

        img = self.normalize(img)
        # img.close()
        return img, int(label.rstrip())

    def __len__(self):
        return len(self.images)

