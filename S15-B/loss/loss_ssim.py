from pytorch_msssim import ssim
"""
The method below uses pytorch implementation of SSIM.
Arg1: Image of 4d tensors (y_gt)
Arg2: Image of 4d tensors (y_pred)
Arg3: Data range of the images. Default is 0-1
Arg4: Average sizing boolean
"""
def get_ssim_loss(img1, img2, data_range=1,size_average=False):
    ssim_val = ssim(img1, img2, data_range=1, size_average=False)
    loss_ssim = (1-ssim_val)/2
    print("SSIM of the inputs =======> {} with Loss as =======> {}".format(ssim_val.item(),loss_ssim.item()))
    return loss_ssim