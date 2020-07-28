## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        #  
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        # Convolutional Layer 1
        # 1 input, 32 outputs, 5x5 filter kernel
        # Input image size: 224x224
        # Output Width = (Width - Filter_Width + 2*Padding)/Stride + 1
        # Output Width = (224 - 5 + 2*0)/1 + 1 = 220
        # Output image size: 220x220
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)

        # Maxpool layer, size 2, stride 2, will be used after all conv layers
        # Output Image size: 110x110
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Convolutional Layer 2
        # Input image size: 110x110
        # Output Width = (Width - Filter_Width + 2*Padding)/Stride + 1
        # Output Width = (110 - 5 + 2*0)/1 + 1 = 106
        # Output image size: 106x106
        # Image size after 2,2 Maxpooling: 53x53
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5)

        # Convolutional Layer 3
        # Input image size: 53x53
        # Output Width = (Width - Filter_Width + 2*Padding)/Stride + 1
        # Output Width = (53 - 5 + 2*0)/1 + 1 = 49
        # Output image size: 49*49
        # Image size after 2,2 Maxpooling: 24x24
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5)

        # Convolutional Layer 4
        # Input image size: 24x24
        # Output Width = (Width - Filter_Width + 2*Padding)/Stride + 1
        # Output Width = (24 - 5 + 2*0)/1 + 1 = 20
        # Output image size: 10*20
        # Image size after 2,2 Maxpooling: 10x10
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 5)

        # Dropouts to prevent over-fitting
        self.dropC = nn.Dropout(p=0.4)
        self.dropL = nn.Dropout(p=0.25)

        # Fully Connected (Linear) Layers
        # Start with 10x10 x 256 = 25,600 features
        # End with 2 x 68 = 136 features
        self.fc1 = nn.Linear(in_features = 256*10*10, out_features = 256*4)
        self.fc2 = nn.Linear(in_features = 256*4, out_features = 256*2)
        self.final = nn.Linear(in_features = 256*2, out_features = 68*2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # 4 convolution + ReLU + Max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropC(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropC(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropC(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropC(x)
        
        # Prepare for Linear layer
        x = x.view(x.size(0), -1)
        
        # Fully connected Layers: Linear -> Dropout -> Linear -> Dropout -> Linear (Final)
        x = F.relu(self.fc1(x))
        x = self.dropL(x)
        x = F.relu(self.fc2(x))
        x = self.dropL(x)
        x = self.final(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
