from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import store_images
from loss.loss import get_loss

train_losses = []
train_acc = []
output_images_depth = []
output_images_mask = []


def train(model, device, trainloader, epoch, optimizer,  criterion=None):
    running_loss = 0.0
    correct = 0
    processed = 0
    if criterion is None:
      criterion =  nn.BCELoss()
    criterion = criterion
    pbar = tqdm(trainloader)
    for i, data in enumerate(pbar):
          inputs, labels = data
          overlay = inputs[1]
          bg = inputs[0]
          mask = labels[0]
          depth = labels[1]
        # Adding bg with overlay
          input_depth_mask = bg + overlay
            
          # Coverting to run on GPU/CPU
          input_s, depth_gt, mask_gt = input_depth_mask.to(device), depth.to(device), mask.to(device)
            
          # zero the parameter gradients
          optimizer.zero_grad()
            
          output_depth, output_mask = model(input_s)
          output_depth, output_mask = output_depth.to(device), output_mask.to(device)
            
          # Appending depth train output to display 
          output_images_depth.append(output_depth)
            
          # Appending mask train output to display 
          output_images_mask.append(output_mask)
            
          # Loss Computations 
          loss_mask = criterion(output_mask, mask_gt)
          loss_depth, ssim_val = get_loss(output_depth, depth_gt)
          loss = loss_depth.sum() + loss_mask
          #print("Total loss: {} sum of Depth loss : {}, SSIM : {} & Mask loss: {}".format(loss,loss_depth,ssim_val,loss_mask))
        
          train_losses.append(loss) 
      
          loss.backward()  # One can set retain graph if there are other losses to accumulate separately (retain_graph=True)
 
          optimizer.step()

          # Save the model, optimiser and epoch at every 100 parameters
          if i%100==0:
            torch.save({'epoch':epoch,
                'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()
              }, 'Depth_Mask_ModelV6_56_240k_250k_v2-local-transform.pt')
            
          f1_score = dice_coef(mask_gt,output_mask)
          mean_acc_depth_mask = (ssim_val+f1_score)/2
          train_acc.append(mean_acc_depth_mask)
          pbar.set_description(desc= f'Loss={loss.item()} Batch_id={i} Accuracy={100*mean_acc_depth_mask:0.2f}')


        
print('Finished Training')
'''
Dice Coffiecient accuracy in segmentation.
There are two recommended ways avaialble : F-1 Dice-Coef and IOU.
'''
def dice_coef(y_true, y_pred, smooth=1):
    mul = y_true * y_pred
    intersection = mul.sum()
    union = y_true.sum() + y_pred.sum()
    dice = torch.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice