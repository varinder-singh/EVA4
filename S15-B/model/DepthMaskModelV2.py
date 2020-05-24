import torch
import torch.nn as nn

'''
Encoder model of main model - feature extractor
'''
class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
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
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU())
        
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
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1), padding=0, stride=1, bias=False))
        self.mode = mode
        self.scale_factor = scale
        
    def forward(self,x):
        out = self.interpolate(x, size=x.shape[-1]*self.scale_factor, mode=self.mode, align_corners=False)
        out = self.conv1(out)
        return out
    
    
'''
Main model

'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1_encoder = Encoder(3,32)
        self.layer2_encoder = Encoder(32,64)
        self.layer3_encoder = Encoder(64,128)
        self.layer4_encoder = Encoder(128,256)
        self.layer5_encoder = Encoder(256,512)
        
        self.layer_maxpool = nn.MaxPool2d(2, 2)
        
        self.layer1_decoder = Decoder(512,512)
        self.layer2_decoder = Decoder(256,256)
        self.layer3_decoder = Decoder(128,128)
        self.layer4_decoder = Decoder(64,64)
        self.layer5_decoder = Decoder(32,32)
        
        self.layer1_interpolate = Interpolate(in_ch=512, out_ch=256, scale=2, mode='bilinear')
        self.layer2_interpolate = Interpolate(in_ch=256, out_ch=128, scale=2, mode='bilinear')
        self.layer3_interpolate = Interpolate(in_ch=128, out_ch=64, scale=2, mode='bilinear')
        self.layer4_interpolate = Interpolate(in_ch=64, out_ch=32, scale=2, mode='bilinear')
        self.layer5_interpolate = Interpolate(in_ch=32, out_ch=1, scale=2, mode='bilinear')
        
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1), padding=0, stride=1, bias=False),
        nn.ReLU(),    
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1, stride=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(0.15),
        nn.Conv2d(in_channels=128, out_channels=1, kernel_size=(1,1), padding=0, stride=1, bias=False)
        )
        
    def forward(self,x):
        # Encoding layers
        out_enc_1 = self.layer1_encoder(x)
        out_enc_mx_1 = self.layer_maxpool(out_enc_1) # IN:224x224x3   OUT:112x112x32
        
        out_enc_2 = self.layer2_encoder(out_enc_mx_1)
        out_enc_mx_2 = self.layer_maxpool(out_enc_2) # IN:112x112x32  OUT:56x56x64
        
        out_enc_3 = self.layer3_encoder(out_enc_mx_2)
        out_enc_mx_3 = self.layer_maxpool(out_enc_3) # IN:56x56x64    OUT:28x28x128
        
        out_enc_4 = self.layer4_encoder(out_enc_mx_3)
        out_enc_mx_4 = self.layer_maxpool(out_enc_4) # IN:28x28x128   OUT:14x14x256
        
        out_enc_5 = self.layer5_encoder(out_enc_mx_4) 
        out_enc_mx_5 = self.layer_maxpool(out_enc_5) # IN:14x14x256   OUT:7x7x512
        
         # Decoding layers
        out_dec_1 = self.layer1_decoder(out_enc_mx_5) 
        out_interp_1 = self.layer1_interpolate(out_dec_1) # IN:7x7x512 OUT:14x14x256
        out_dec_enc_res_interp_1 = out_interp_1 + out_enc_mx_4
        
        out_dec_2 = self.layer2_decoder(out_dec_enc_res_interp_1)
        out_interp_2 = self.layer2_interpolate(out_dec_2) # IN:14x14x256 OUT:28x28x128
        out_dec_enc_res_interp_2 = out_interp_2 + out_enc_mx_3
        
        out_dec_3 = self.layer3_decoder(out_dec_enc_res_interp_2)
        out_interp_3 = self.layer3_interpolate(out_dec_3) # IN:28x28x128 OUT:56x56x64
        out_dec_enc_res_interp_3 = out_interp_3 + out_enc_mx_2
        
        out_dec_4 = self.layer4_decoder(out_dec_enc_res_interp_3)
        out_interp_4 = self.layer4_interpolate(out_dec_4) # IN:56x56x64 OUT:112x112x32
        out_dec_enc_res_interp_4 = out_interp_4 + out_enc_mx_1
        
        out_dec_5 = self.layer5_decoder(out_dec_enc_res_interp_4)
        out_interp_5 = self.layer5_interpolate(out_dec_5) # IN:56x56x64 OUT:112x112x32
        
        out_mask = self.conv1(out_interp_5)
        out_depth = out_interp_5
        
        return out_depth, out_mask