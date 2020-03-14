import torch.nn as nn
import torch.nn.functional as F

dropout_value=0.20
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ############################### Convolution Block 1 ###############################
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=2, dilation=2, bias=False),# Atrous Convolution
                                   nn.ReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(32),
                                   nn.Dropout2d(dropout_value))# Input=32x32x3 Output=32x32x64 RF=7x7
        ############################## Transition Block 1 ##################################
        self.pool1 = nn.MaxPool2d(2, 2)# Input=32x32x64 Output=16x16x64 RF=8x8
        
        
        
         ############################### Convolution Block 2 ###############################
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(dropout_value))# Input=16x16x64 Output=16x16x128 RF=16x16
        ############################## Transition Block 1 ##################################
        self.pool2 = nn.MaxPool2d(2, 2)# Input=16x16x64 Output=8x8x128 RF=18x18
        
        
        
         ############################### Convolution Block 3 ###############################
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Dropout2d(dropout_value))# Input=8x8x128 Output=8x8x256 RF=34x34
        ############################## Transition Block 1 ##################################
        self.pool3 = nn.MaxPool2d(2, 2)# Input=8x8x128 Output=4x4x128 RF=38x38
        
        
        
         ############################### Convolution Block 4 ###############################
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1, groups=128, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(128),
                                   nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), padding=0, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(256),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1, groups=256, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(256),
                                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), padding=0, bias=False),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(512),
                                   nn.Dropout2d(dropout_value))# Input=4x4x256 Output=4x4x512 RF=60x60
        
        ############################## GAP Layer ##################################
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))# Input=4x4x512 Output=1x1x512 RF=84x84
        
        ############################# 1x1 Layer ###################################
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1,1), padding=0, bias=False))


    def forward(self, x):
      x = self.pool1(self.conv1(x))
      x = self.pool2(self.conv2(x))
      x = self.pool3(self.conv3(x))
      x = self.conv4(x)
      x = self.gap(x)
      x = self.conv5(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)

      
net=Net()