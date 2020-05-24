import torch
import torch.nn.functional as F
from loss.loss import get_loss

test_losses = []
test_acc = []
output_images_depth = []
output_images_mask = []

def test(model, device, testloader, criterion=None):
    correct = 0
    total = 0
    test_loss = 0
    mean_acc_depth_mask = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            overlay = inputs[1]
            bg = inputs[0]
            mask = labels[0]
            depth = labels[1]
            # Adding bg with overlay
            input_depth_mask = bg + overlay
            
           # Coverting to run on GPU/CPU
            input_s, depth_gt, mask_gt = input_depth_mask.to(device), depth.to(device), mask.to(device)
            
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
            #print("Total loss: {} sum of Depth loss :{} Mask loss: ".format(loss,loss_depth,loss_mask))
            
            
            test_loss += loss
            f1_score = dice_coef(mask_gt,output_mask)
            mean_acc_depth_mask += (ssim_val+f1_score)/2
            
    test_losses.append(test_loss)
   
    print('\nTest Set: Average loss: {:.2f} , Accuracy: ({:.2f}%)\n'.format(test_loss/len(testloader), mean_acc_depth_mask / len(testloader)))
    
    
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