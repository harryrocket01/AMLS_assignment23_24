
import numpy as np
from numpy.typing import ArrayLike


import torch
import torchvision
import torchvision.transforms as transforms
torch.manual_seed(42)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable

#Sets type of torch to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


import pandas as pd

class ML_Template():

    def __init__(self,X_train,y_train,X_val,y_val,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

class MV_Models(ML_Template):

    def CNN(self,batch_size=128,epochs=25,learning_rate=0.01):
        """
        Test implamentation of base CNN based off Pytorch Documentation
        """

        X_train = torch.tensor(self.X_train.transpose(0, 3, 1, 2), dtype=torch.float32)
        y_train = torch.tensor(self.y_train, dtype=torch.long)
        X_val = torch.tensor(self.X_val.transpose(0, 3, 1, 2), dtype=torch.float32)
        y_val = torch.tensor(self.y_val, dtype=torch.long)
        X_test = torch.tensor(self.X_test.transpose(0, 3, 1, 2), dtype=torch.float32)
        y_test = torch.tensor(self.y_test, dtype=torch.long)
        
        if y_train.dim() > 1:
            y_train = y_train.squeeze()
        if y_val.dim() > 1:
            y_val = y_val.squeeze()    
        if y_test.dim() > 1:
            y_test = y_test.squeeze()
  

        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)
        test_data = TensorDataset(X_test, y_test)


        batch_size = 10
        loaders = {
            'train' : DataLoader(train_data, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=1),
            'val'  : DataLoader(val_data, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=1),            
            'test'  : DataLoader(test_data, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=1),
        }       

        model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2]).to(device)
        loss_func = nn.CrossEntropyLoss()   
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)   

        print("Ready To Train")
        for epoch in range(epochs):
            model.train()  # Set the model to training mode
            for images, labels in loaders['train']:
                images, labels = images.to(device), labels.to(device)
                # Forward pass
                outputs = model(images)
                loss = loss_func(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
   
            val_accuracy = self.accuracy_model(model, loaders['val'], device)

            print('Epoch [{}/{}], Loss: {:.4f}, Validation Accuracy: {:.2f}%'
                .format(epoch + 1, epochs, loss.item(), val_accuracy))

        # Test the model
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in loaders['test']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
            
        test_accuracy = 100 * test_correct / test_total
        print('Test Accuracy: {:.2f}%'.format(test_accuracy))
            
    def accuracy_model(model, data_loader, device):
        model.eval()  # Set the model to evaluation mode
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        return val_accuracy  
        
        
    


"""
in_channels=1: because our input is a grayscale image.

Stride: is the number of pixels to pass at a time when sliding the convolutional kernel.

Padding: to preserve exactly the size of the input image, it is useful to add a zero padding on the border of the image.

kernel_size: we need to define a kernel which is a small matrix of size 5 * 5. To perform the convolution operation, we just need to slide the kernel along the image horizontally and vertically and do the dot product of the kernel and the small portion of the image.

The forward() pass defines the way we compute our output using the given layers and functions.

"""
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   
            nn.BatchNorm2d(16) # Example of batch normalization
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32) # Example of batch normalization
        )
        self.out = nn.Linear(32 * 7 * 7, 10) # Adjust if input size is different

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28*3, 64)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28*3)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)
    
















#RESNET




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        #x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return probas




