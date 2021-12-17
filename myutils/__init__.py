#from utils import *
import os
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
from math import log10
from skimage.metrics import structural_similarity as ssim


def to_psnr(output, gt):
    mse = F.mse_loss(output, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    # import pdb; pdb.set_trace()
    return psnr_list

def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)
    dehaze_list_np = [dehaze_list[ind].data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list

