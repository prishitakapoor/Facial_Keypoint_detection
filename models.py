## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        # after one pool layer, this becomes (32, 111, 111)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (111-3)/1 +1 = 109
        # the output tensor will have dimensions: (64, 109, 109)
        # after another pool layer this becomes (20, 54, 54);
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        # third conv layer: 20 inputs, 40 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (128, 52, 52)
        # after another pool layer this becomes (128, 26, 26);
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        # forth conv layer: 20 inputs, 40 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        # the output tensor will have dimensions: (256, 24, 24)
        # after another pool layer this becomes (256, 12, 12);
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(256)
        # after another pool layer this becomes (512, 5, 5);
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.bn5 = nn.BatchNorm2d(512)
        # 20 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(512*5*5, 4000)
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc2_drop = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 136)
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))
        x = F.relu(self.pool3(self.bn3(self.conv3(x))))
        x = F.relu(self.pool4(self.bn4(self.conv4(x))))
        x = F.relu(self.pool5(self.bn5(self.conv5(x))))
        # print(x.shape)
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # two linear layers with dropout in between
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    
class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        n_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(n_inputs, 136)
        self.resnet50.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
                        
    def forward(self, x):
        x = self.resnet50(x)
        return x
    
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet18(x)
        return x
    

class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16.classifier[6] = nn.Linear(4096,136)
                        
    def forward(self, x):
        x =self.vgg16(x)
        return x