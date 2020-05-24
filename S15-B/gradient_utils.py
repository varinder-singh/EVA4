import torch.nn.functional as F
import torch
import numpy as np

def gradient(img):
    padded_tensor = F.pad(img, (0,1,0,1), mode='replicate')
    pad_np = padded_tensor.detach().cpu().numpy()
    grad = _grad_dx_dy(pad_np)
    grad_no_pad = grad[:,:,:-1,:-1]
    return torch.from_numpy(grad_no_pad)



def _grad_dx_dy(img):
    for b in range(img.shape[0]):
        for i in range(img.shape[1]):
            for j in range(img.shape[2]):
                for k in range(img.shape[3]):
                    curr_pixel_val = img[b][i][j][k-1]
                    next_pixel_val = img[b][i][j][k]
                    new_pixel = np.abs(next_pixel_val - curr_pixel_val)
                    #print("Current px {},\tNext px {},\tabsolute diff px {}".format(curr_pixel_val,next_pixel_val,new_pixel))
                    img[b][i][j][k] = new_pixel
    return _grad_dy(img)

def _grad_dy(img):
    grad_img_dy = img
    for b in range(img.shape[0]):
        for i in range(grad_img_dy.shape[1]):
            for j in range(grad_img_dy.shape[2]):
                for k in range(grad_img_dy.shape[3]):
                    curr_pixel_val = grad_img_dy[b][i][j-1][k]
                    next_pixel_val = grad_img_dy[b][i][j][k]
                    new_pixel = np.abs(next_pixel_val - curr_pixel_val)
                    #print("Current px {},\tNext px {},\tabsolute diff px {}".format(curr_pixel_val,next_pixel_val,new_pixel))
                    grad_img_dy[b][i][j][k]=new_pixel
    return grad_img_dy

