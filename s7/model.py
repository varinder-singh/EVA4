import torch.nn as nn
import torch.nn.functional as F

dropout_value=0.25
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ############################### Convolution Block 1 ###############################
        self.conv1_dilated = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=2, dilation=2, bias=False),
                                   nn.ReLU(),
                                  nn.BatchNorm2d(64),
                                  nn.Dropout2d(dropout_value))# Input=32x32x3 Output=32x32x16 RF=5x5
        ############################## Transition Block 1 ##################################
        self.pool1 = nn.MaxPool2d(2, 2)
        
        
        
         ############################### Convolution Block 2 ###############################
        self.conv2_depthwise_sep = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1,
                                                           groups=64,bias=False),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), padding=0, bias=False),
                                   nn.ReLU(),             
                                  nn.BatchNorm2d(128),
                                  nn.Dropout2d(dropout_value))# Input=16x16x16 Output=16x16x32 RF=10x10
        ############################## Transition Block 1 ##################################
        self.pool2 = nn.MaxPool2d(2, 2)
        
        
        
         ############################### Convolution Block 3 ###############################
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1, bias=False),
                                   nn.ReLU(),
                                  nn.BatchNorm2d(256),
                                  nn.Dropout2d(dropout_value))# Input=8x8x32 Output=8x8x64 RF=20x20
        ############################## Transition Block 1 ##################################
        self.pool3 = nn.MaxPool2d(2, 2)
        
        
        
         ############################### Convolution Block 4 ###############################
        self.conv4_depthwise_sep = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1, groups=256, bias=False),
                                                  nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1,1), padding=0, bias=False),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(512),
                                  nn.Dropout2d(dropout_value))# Input=4x4x64 Output=4x4x128 RF=36x36
        
        ############################## GAP Layer ##################################
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))# Input=4x4x128 Output=1x1x128 RF=64x64
        
        ############################# 1x1 Layer ###################################
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1,1), padding=0, bias=False))

    def forward(self, x):
        x = self.pool(self.conv1_dilated(x))
        x = self.pool(self.conv2_depthwise_sep(x))
        x = self.pool(self.conv3(x))
        x = self.conv4_depthwise_sep(x)
        x = self.gap(x)
        x = self.conv5(x)
        x = x.view(-1, 10)
        return x


