from loss.loss_point_wise import get_pointwise_loss
from loss.loss_gradient import get_grad_loss
from loss.loss_ssim import get_ssim_loss

"""
The method below returns point-wise loss for given two images as an argument.
Arg1: Image of 4d tensors (y_pred)
Arg2: Image of 4d tensors (y_gt)
return: loss od Depth
"""
def get_loss(output, label):
    lambda_1 = 0.1
    loss = lambda_1 * get_pointwise_loss(output, label) + get_grad_loss(output, label) + get_ssim_loss(output, label)
    print("Loss in Dense Depth is =======> {}".format(loss.item()))
    return loss
    