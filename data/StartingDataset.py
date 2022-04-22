import torch
from torchvision import transforms
from PIL import Image


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        self.images = []
        with open("cassava-leaf-disease-classification/train.csv", "r") as f:
            tensor_converter = transforms.ToTensor()
            for line in f[1:7]:
                elems = line.split(',')
                file = elems[0]
                label = elems[1]
                img = Image.open("../cassava-leaf-disease-classification/train_images/" + file)
                tensor = tensor_converter(img)
                self.images.append((tensor, label))
        

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

