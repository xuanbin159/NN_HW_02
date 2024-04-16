import tarfile
from PIL import Image
import os
import torch
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.1307], [0.3081])])

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
                - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.file_open = tarfile.open(data_dir, 'r')
        self.png_files = self.file_open.getnames()[1:]
        self.labels = [int(os.path.basename(i)[-5]) for i in self.png_files]
        self.numbers = [os.path.basename(i)[:5] for i in self.png_files]
        self.transform = transform
        self.is_train = is_train
    def __len__(self):

        return len(self.numbers)

    def __getitem__(self, idx):

        str_idx = str(idx)

        if len(str(idx)) != 5:
            str_idx = '0' * (5 - len(str_idx)) + str_idx

        label = torch.tensor(self.labels[self.numbers.index(str_idx)])

        if self.is_train:
            folder = 'train'
        else:
            folder = 'test'

        extract_file = self.file_open.extractfile(f'{folder}/{str_idx}_{label}.png').read()
        open_image = Image.open(io.BytesIO(extract_file))
        img = transform(open_image)

        return img, label


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.1307], [0.3081])])

    train = MNIST(data_dir='/content/drive/MyDrive/Colab Notebooks/mnist-classification/data/train.tar', transform=transform, is_train=True)
    test = MNIST(data_dir='/content/drive/MyDrive/Colab Notebooks/mnist-classification/data/test.tar', transform=transform, is_train=False)
    train_data = DataLoader(train, batch_size=64)
    test_data = DataLoader(test, batch_size=64)

    print(test.__getitem__(0)[1])
    # tensor(6)
    print(train.__getitem__(0)[1])
    # tensor(5)

    # print dataloader size
    print(next(iter(train_data))[0].size())
    # torch.Size([64, 1, 28, 28])