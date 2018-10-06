## TODO: define the convolutional neural network architecture

import torch
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
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1_1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
      
        #input for conv2 64 112 112
        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)

        #input for conv3 128 56 56
        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        #input for conv4 256 28 28
        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        #input for conv5 256 14 14
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512*7*7,4096)
#         dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(4096, 136)
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step
        x = self.pool(F.relu(self.conv1_1(x)))     
        x = self.pool(F.relu(self.conv2_1(x)))     
        x = self.pool(F.relu(self.conv3_1(x)))
        x = self.pool(F.relu(self.conv4_1(x)))
        x = self.pool(F.relu(self.conv5_1(x)))
        
        x = x.view(x.size(0), -1)
        x = self.fc1_drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
