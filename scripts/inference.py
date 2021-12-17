from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
from torch.autograd import Variable
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from misc import *
import models.mopnet as net
from models.vgg16 import Vgg16
from myutils import utils
from visualizer import Visualizer
import time

from skimage.metrics import peak_signal_noise_ratio as Psnr#measure import compare_psnr as Psnr
from skimage.metrics import structural_similarity as ssim

import torch.nn.functional as F
import scipy.stats as st
import datetime

from PIL import Image
import math
import numpy as np
import cv2
from collections import OrderedDict

from PIL import Image
import torchvision.transforms as transforms

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='my_loader',  help='')
parser.add_argument('--dataroot', required=False,
  default='./data/custom-data', help='path to trn dataset')
parser.add_argument('--netG', default='mopnet/netG_epoch_150.pth', help="path to netG (to continue training)")
parser.add_argument('--netE', default="mopnet/netEdge_epoch_150.pth", help="path to netE (to continue training)")
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=532, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=256, help='the height / width of the cropped input image to network')
parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
parser.add_argument('--image_path', type=str, default='results', help='path to save the generated vali image')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--record', type=str, default='default.txt', help='prefix of different dataset')
parser.add_argument('--number', type=int, default=10)
parser.add_argument('--write', type=int, default=1, help='if write the results?')
opt = parser.parse_args()
print(opt)

path_class_color = "../classifier/color_epoch_95.pth"
path_class_geo = "../classifier/geo_epoch_95.pth"

device = torch.device("cuda:0")

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# val_dataloader = getLoader(opt.dataset,
#                        opt.dataroot,
#                        opt.originalSize,
#                        opt.imageSize,
#                        opt.batchSize,
#                        opt.workers,
#                        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
#                        split='test',
#                        shuffle=False,
#                        seed=opt.manualSeed,
#                        pre=opt.pre)

inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

netG=net.Single()
netG.load_state_dict(torch.load(opt.netG))
netG.eval()
netG.to(device)
netEdge = net.EdgePredict()
netEdge.load_state_dict(torch.load(opt.netE))
netEdge.eval()
netEdge.to(device)
print(netG)


imageWidth=1080#int(1920/2)
imageHeight=720#int(1080/2)

target = torch.FloatTensor(opt.batchSize, outputChannelSize, imageHeight, imageWidth)#opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, imageHeight, imageWidth)#opt.imageSize, opt.imageSize)
target, input = target.to(device), input.to(device)

# Classifiers
net_label_color=net.vgg19ca()
net_label_color.load_state_dict(torch.load(path_class_color))
net_label_color=net_label_color.to(device)

net_label_geo = net.vgg19ca_2()
net_label_geo.load_state_dict(torch.load(path_class_geo))
net_label_geo=net_label_geo.to(device)

vcnt = 0

# Sobel kernel Conv
a = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)
a = a.reshape(1, 1, 3, 3)
a = np.repeat(a, 3, axis=0)
conv1=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
conv1.weight.data.copy_(torch.from_numpy(a))
conv1.weight.requires_grad = False
conv1.cuda()

b = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=np.float32)
b = b.reshape(1, 1, 3, 3)
b = np.repeat(b, 3, axis=0)
conv2=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
conv2.weight.data.copy_(torch.from_numpy(b))
conv2.weight.requires_grad = False
conv2.cuda()

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    return out_filter

# Gaussian blur
g_kernel = gauss_kernel(3, 5, 1).transpose((3, 2, 1, 0))
gauss_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=3/2, bias=False)
gauss_conv.weight.data.copy_(torch.from_numpy(g_kernel))
gauss_conv.weight.requires_grad = False
gauss_conv.cuda()

#input_path_inference = './data/training-data'
import os

mean = ((0.5,0.5,0.5))
std = ((0.5,0.5,0.5))
#imageSize = 1024
transform = transforms.Compose([ 
  #transforms.Resize(imageHeight,imageWidth),
  transforms.CenterCrop(opt.imageSize),
  transforms.ToTensor(),
  transforms.Normalize(mean, std),])


img_path = opt.image_path
#utils.mkdirs([os.path.join(img_path,c) for c in ['d','o','g']])

i = 0
vpsnr=0
vssim=0
vcnt=0
for file in os.listdir(opt.dataroot):
  
  path = os.path.join(opt.dataroot,file)
  img = Image.open(path).convert('RGB')
  file_target = file.split('_')[-1]
  path_gt='{}/{}'.format(opt.dataroot.replace('source','target'),file.replace('source','target'))
  #path_gt=os.path.join('../data_1858_100_100/test/target','sourceImage_'+file_target)
  img_gt=Image.open(path_gt).convert('RGB')
  w, h = img.size
  
  # not sure why we need this....
  #region = img.crop((1+int(0.15*w), 1+int(0.15*h), int(0.85*w), int(0.85*h)))

  input = transform(img)
  input = input.float().to(device)
  target = transform(img_gt)
  target = target.float().to(device)

  with torch.no_grad():

    #import pdb; pdb.set_trace()
    input.resize_as_(input).copy_(input)
    input = input.unsqueeze(0)
    target.resize_as_(target).copy_(target)
    target = target.unsqueeze(0)

    i_G_x = conv1(input)
    i_G_y = conv2(input)
    iG = torch.tanh(torch.abs(i_G_x)+torch.abs(i_G_y))

    #import pdb; pdb.set_trace();
    # predict color labels
    _, label_color = torch.max(net_label_color(input), 1)
    label_curve, label_thick = net_label_geo(iG)
    _, label_curve = torch.max(label_curve, 1)
    _, label_thick = torch.max(label_thick, 1)
    label_curve = label_curve.float()
    label_color = label_color.float()
    label_thick = label_thick.float()
    labels = [label_curve, label_color, label_thick]

    #import pdb; pdb.set_trace();
    # Get input edges
    i_G_x_ = conv1(input)
    i_G_y_ = conv2(input)
    input_edge = torch.tanh(torch.abs(i_G_x_)+torch.abs(i_G_y_))


    #import pdb; pdb.set_trace();
    # Get predicted edges
    edge1 = netEdge(torch.cat([input, input_edge], 1))
    _, edge = edge1
    input.cuda()
    edge.cuda()
    
    #import pdb; pdb.set_trace();
    # Moire removal
    x_hat1 = netG(input, edge, labels)
    residual, x_hat = x_hat1

    # Save results
    for j in range(x_hat.shape[0]):
        vcnt += 1
        b, c, w, h = x_hat.shape
        ti1 = x_hat[j, :,:,: ]
        tt1 = target[j, :,:,: ]
        ori = input[j, :, :, :]


        mi1 = cv2.cvtColor(utils.my_tensor2im(ti1), cv2.COLOR_BGR2RGB)
        mt1 = cv2.cvtColor(utils.my_tensor2im(tt1), cv2.COLOR_BGR2RGB)
        ori = cv2.cvtColor(utils.my_tensor2im(ori), cv2.COLOR_BGR2RGB)

        vpsnr+= Psnr(mt1,mi1)
        vssim+=ssim(mi1,mt1,multichannel=True)

        if opt.write==1:
            cv2.imwrite(img_path + os.sep+'d'+os.sep+file+'.png', mi1)
            cv2.imwrite(img_path + os.sep+'o'+os.sep+file+'.png',ori)
            cv2.imwrite(img_path + os.sep+'g'+os.sep+file+'.png',mt1)
            #cv2.imwrite(opt.image_path + os.sep + 'g'+os.sep+'g'+'_'+str(i) +'_.png', mt1)
    print(50*'-')
    print(vcnt)
    print(50*'-')


avr_psnr = float(vpsnr)/vcnt#sum(psnr_list) / len(psnr_list)
avr_ssim = float(vssim)/vcnt#sum(ssim_list) / len(ssim_list)
print('psnr : {:>10f}, ssim : {:>10f} '.format(avr_psnr, avr_ssim))
print(50*'-')
          