import torch

"""
The method below returns point-wise loss for given two images as an argument.
Arg1: Image of 4d tensors (y_gt)
Arg2: Image of 4d tensors (y_pred)
return: point wise loss
"""
def get_pointwise_loss(img1, img2):
    loss_pointwise = torch.mean(torch.abs(img1-img2))
    print("Point Wise loss for GT and Pred images =======> {}".format(loss_pointwise.item()))
    return loss_pointwise