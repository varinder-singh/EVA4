import torch
import torch.nn as nn

'''
Encoder model of main model - feature extractor
'''
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.dropout = 0.1
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch*2, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch*2),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*4, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch*4),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=out_ch*4, out_channels=out_ch, kernel_size=(1,1), padding=0, stride=1, bias=False)
        )
        
    def forward(self,x):
        out = self.conv1(x)
        return out

'''
Decoder model of main model
'''
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder,self).__init__()
        self.dropout = 0.1
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch//2, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch//2),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=out_ch//2, out_channels=out_ch//4, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch//4),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=out_ch//4, out_channels=out_ch, kernel_size=(1,1), padding=0, stride=1, bias=False)
        )
        
    def forward(self,x):
        out = self.conv1(x)
        return out
        
'''
Interpolater - Binary Interpolate
'''
class Interpolate(nn.Module):
    def __init__(self, in_ch, out_ch, scale, mode):
        super(Interpolate, self).__init__()
        self.interpolate = nn.functional.interpolate
        self.mode = mode
        self.scale_factor = scale
        
    def forward(self,x):
        out = self.interpolate(x, size=x.shape[-1]*self.scale_factor, mode=self.mode, align_corners=False)
        return out
    
    
'''
Main model

'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout = 0.1

        self.layer1_encoder = Encoder(32,64)
        self.layer2_encoder = Encoder(64,128)
        self.layer3_encoder = Encoder(128,256)
        
        self.layer_maxpool = nn.MaxPool2d(2, 2)
        
    
        self.layer1_decoder = Decoder(256,128)
        self.layer2_decoder = Decoder(128,64)
        self.layer3_decoder = Decoder(64,32)
 
        
   
        self.layer1_interpolate = Interpolate(in_ch=128, out_ch=64, scale=2, mode='bilinear')
        self.layer2_interpolate = Interpolate(in_ch=64, out_ch=32, scale=2, mode='bilinear')
        self.layer3_interpolate = Interpolate(in_ch=32, out_ch=1, scale=2, mode='bilinear')
        
        # This is the first layer to recive the image 
        self.con0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU())
        
        
        self.conv_mask = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(self.dropout),
        nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1,1), padding=0, stride=1, bias=False)
        )
        
        self.conv_depth = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,1), padding=0, stride=1, bias=False)
        )
        
        
    def forward(self,x):
        # Encoding layers
        
        out_conv_0 = self.con0(x) # IN: 56x56x3 OUT: 56x56x32
        
        out_enc_1 = self.layer1_encoder(out_conv_0)
        out_enc_mx_1 = self.layer_maxpool(out_enc_1) # IN:56x56x32    OUT:28x28x64
        
        out_enc_2 = self.layer2_encoder(out_enc_mx_1)
        out_enc_mx_2 = self.layer_maxpool(out_enc_2) # IN:28x28x64   OUT:14x14x128
        
        out_enc_3 = self.layer3_encoder(out_enc_mx_2) 
        out_enc_mx_3 = self.layer_maxpool(out_enc_3) # IN:14x14x128   OUT:7x7x256
        
        
       
         # Decoding layers
             
        out_dec_1 = self.layer1_decoder(out_enc_mx_3)
        out_interp_1 = self.layer1_interpolate(out_dec_1) # IN: 7x7x256  OUT:14x14x128
        out_dec_enc_res_interp_1 = out_interp_1 + out_enc_mx_2
        
        out_dec_2 = self.layer2_decoder(out_dec_enc_res_interp_1)
        out_interp_2 = self.layer2_interpolate(out_dec_2) # IN: 14x14x128  OUT:28x28x64
        out_dec_enc_res_interp_2 = out_interp_2 + out_enc_mx_1
        
        out_dec_3 = self.layer3_decoder(out_dec_enc_res_interp_2)
        out_interp_3 = self.layer3_interpolate(out_dec_3) # IN: 28x28x64  OUT:56x56x32
        
        
        
        out_mask = self.conv_mask(out_interp_3)
        out_depth = self.conv_depth(out_interp_3)
        
        return out_depth, out_mask