import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv_layers = nn.Sequential(           
            # C1 (5*5*1+1)*6 = 156
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),
            nn.Tanh(),
            # S2
            nn.MaxPool2d(kernel_size=2,stride=2),
            # C3 (5*5*6+1)*16 = 2416
            nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5,stride=1, padding=0, bias=True),
            nn.Tanh(),
            # S4
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.fc_layers = nn.Sequential(
            # C5 (16*5*5+1)*120 = 48120
            nn.Linear(in_features=16 * 5 * 5,out_features=120),
            nn.Tanh(),
            # F6 (120+1)*84 = 10164
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            # OUTPUT (84+1)*10 = 850
            nn.Linear(in_features=84,out_features=10),
            nn.Softmax(dim=1)
        )

       # Total number of parameters = 61,706
       # C1: (5*5*1+1)*6 = 156
       # C3: (5*5*6+1)*16 = 2,416
       # C5: (16*5*5+1)*120 = 48,120
       # F6: (120+1)*84 = 10,164
       # OUTPUT: (84+1)*10 = 850
       # Total: 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706

    def forward(self, img):
        x = self.conv_layers(img)
        x = x.view(-1, 16 * 5 * 5)
        output = self.fc_layers(x)

        return output


class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        # Assuming input images are 28x28, flatten them to 1D vector (28*28 = 784)
        self.fc1 = nn.Linear(28*28, 82)  # First hidden layer 64370
        self.fc2 = nn.Linear(82, 56)     # Second hidden layer 4648
        self.fc3 = nn.Linear(56, 28)      # Third hidden layer 1596
        self.fc4 = nn.Linear(28, 10)      # Output layer, for 10 classes  290
        

    def forward(self, img):
        # Flatten image
        x = img.view(img.size(0), -1)
        
        # Forward pass through fully connected layers with Tanh activations
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        
        # No activation for the last layer as it will output logits
        output = self.fc4(x)
        
        return output

class LeNet52(nn.Module):

    def __init__(self):
        super(LeNet52, self).__init__()

        self.conv_layers = nn.Sequential(
            # C1 (5*5*1+1)*6 = 156
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5,  stride=1,  padding=2,  bias=True),
            nn.Tanh(),
            # S2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # C3 (5*5*6+1)*16 = 2416
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.Dropout(0.2),
            # S4
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            # C5 (16*5*5+1)*120 = 48120
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Tanh(),
            nn.Dropout(0.5),
            # F6 (120+1)*84 = 10164
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Dropout(0.5),
            # OUTPUT (84+1)*10 = 850
            nn.Linear(in_features=84, out_features=10),
            nn.Softmax(dim=1)
        )

       # Total number of parameters = 61,706
       # C1: (5*5*1+1)*6 = 156
       # C3: (5*5*6+1)*16 = 2,416
       # C5: (16*5*5+1)*120 = 48,120
       # F6: (120+1)*84 = 10,164
       # OUTPUT: (84+1)*10 = 850
       # Total: 156 + 2,416 + 48,120 + 10,164 + 850 = 61,706
    def forward(self, img):
        x = self.conv_layers(img)
        x = x.view(-1, 16 * 5 * 5)
        output = self.fc_layers(x)

        return output