import tarfile
from PIL import Image
import os
import torch
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")

class MNIST(Dataset):
    """ MNIST dataset

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
                - Subtract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.file_open = tarfile.open(data_dir, 'r')
        self.images = {}
        self.labels = {}
        self.is_train = is_train
        self.transform = transform
        
        # Extract all images at once and store them in a dictionary
        for member in self.file_open.getmembers():
            if member.name.endswith('.png'):
                number = os.path.basename(member.name)[:5]
                label = int(os.path.basename(member.name)[-5])
                self.labels[number] = label
                file = self.file_open.extractfile(member)
                image = Image.open(io.BytesIO(file.read()))
                self.images[number] = image
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        str_idx = str(idx).zfill(5)
        label = torch.tensor(self.labels[str_idx])
        img = self.images[str_idx]

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.1307], [0.3081])])

    train = MNIST(data_dir='/content/drive/MyDrive/Colab Notebooks/mnist-classification/data/train.tar', transform=transform, is_train=True)
    test = MNIST(data_dir='/content/drive/MyDrive/Colab Notebooks/mnist-classification/data/test.tar', transform=transform, is_train=False)
    train_data = DataLoader(train, batch_size=64)
    test_data = DataLoader(test, batch_size=64)

    print(test.__getitem__(0)[1])  # tensor(6)
    print(train.__getitem__(0)[1])  # tensor(5)
    print(next(iter(train_data))[0].size())  # torch.Size([64, 1, 28, 28])
