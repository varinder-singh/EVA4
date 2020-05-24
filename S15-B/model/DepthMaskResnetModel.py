import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

''' 
The class below does encoding to an input.
The goal is to increase depth and decrease spatial dimensions
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()       
        self.original_model = models.resnet18(pretrained=True )
        self.features = nn.Sequential(*list(self.original_model.children())[:-2])
        

    def forward(self, x):
        x_block0_4 = self.features[:-3](x)
        x_block5 = self.features[5](x_block0_4)
        x_block6 = self.features[6](x_block5)
        x_block7 = self.features[7](x_block6)
        return x_block7
#         for k, v in self.original_model.features._modules.items():
#             print("K {} and V {} ".format(k,v))
#             features.append( v(features[-1]) )
       # return features

''' 
The class below does interpolation to an input 
'''
class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.inter_polate = nn.functional.interpolate
        self.mode = mode
        self.scale_factor = scale
        
    def forward(self, x):
        x = self.inter_polate(x, size=x.shape[-1]*self.scale_factor, mode=self.mode, align_corners=False)
        return x    
''' 
The class below does decoding to an input.
The goal is to decrease depth and increase spatial dimensions
The contructor accepts 3 arguments as the 3 layers ResNet output
'''
# class Decoder(nn.Module):
#     def __init__(self, out1, out2, out3):
#         super(Decoder, self).__init__()
#         self.out1 = out1
#         self.out2 = out2
#         self.out3 = out3
#         self.decoder_layer1 = nn.Sequential(
#             torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=1, bias=False),
#             nn.ReLU()
#         )
#         self.decoder_layer2 = nn.Sequential(
#             torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1), stride=1, bias=False),
#             nn.ReLU()
#         )
#         self.decoder_layer3 =  nn.Sequential(
#            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), stride=1, bias=False),
#             nn.ReLU()
#         )
# #          self.interpolate_layer2 =  nn.Sequential(
# #             torch.nn.Upsample(size=(1025, 15), mode='bilinear')
# #          )
#         self.decoder_layer4 =  nn.Sequential(
#            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=1, bias=False),
#             nn.ReLU()
#         )
#         self.interpolate_layer1 =  nn.Sequential(
#             Interpolate(scale=2, mode='bilinear')
#          )
    
#     def forward(self, x):
#         print("Forward of Decoder is called with input shape is {}".format(x.shape))
#         out = self.decoder_layer1(x)
#         out = self.interpolate_layer1(out)
#         out = out + self.out3 # Concatenating encoder layer3 output with decoder layer 2
#         out = self.decoder_layer2(out)
#         out = self.interpolate_layer1(out)
#         out = out + self.out2 # Concatenating encoder layer2 output with decoder layer 3
#         out = self.decoder_layer3(out)
#         out = self.interpolate_layer1(out)
#         out = out + self.out1 # Concatenating encoder layer1 output with decoder layer 4
#         out = self.decoder_layer4(out)
#         return out

class Decoder(nn.Module):
    def __init__(self, features=2208, decoder_width = 0.5):
        super(Decoder, self).__init__()
        self.encoder_features = features
        self.decoder_layer1 = nn.Sequential(
          torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), padding=1, stride=1, bias=False),
          torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=1, bias=False),
             nn.ReLU()
         )
        self.decoder_layer2 = nn.Sequential(
             torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3),padding=1, stride=1, bias=False),
             torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1,1), stride=1, bias=False),
             nn.ReLU()
         )
        self.decoder_layer3 =  nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),padding=1, stride=1, bias=False),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1,1), stride=1, bias=False),
             nn.ReLU()
         )
        self.decoder_layer4 =  nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),padding=1, stride=1, bias=False),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1,1), stride=1, bias=False),
             nn.ReLU()
         )
        self.interpolate_layer1 =  nn.Sequential(
             Interpolate(scale=2, mode='bilinear')
         )
    
    def forward(self,x):
        x_block7 = x
        out = self.decoder_layer1(x_block7)
        out = self.interpolate_layer1(out) # Upsample
        out = self.interpolate_layer1(out) # Upsample
        out = self.interpolate_layer1(out) # Upsample
        out = self.interpolate_layer1(out) # Upsample
        out = self.interpolate_layer1(out) # Upsample
        out = self.decoder_layer2(out)
        #out = self.interpolate_layer1(out)
        out = self.decoder_layer3(out)
        #out = self.interpolate_layer1(out)
        out = self.decoder_layer4(out)
        return out
    
        
class DepthMaskResnetModel(nn.Module):
    def __init__(self):
        super(DepthMaskResnetModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder( self.encoder(x) )
