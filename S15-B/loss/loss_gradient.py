import torch

"""
The method below returns point-wise loss for given two images as an argument.
Arg1: Image of 4d tensors (y_pred)
Arg2: Image of 4d tensors (y_gt)
return: loss in gradients
"""
def get_grad_loss(img1, img2):
    loss_grad = torch.mean(torch.abs(img1.grad+img2.grad))
    print("Loss gradient for the GT and Pred =======>{}".format(loss_grad.item()))
    return loss_grad