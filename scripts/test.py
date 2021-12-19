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
import torchvision.utils as vutils
from torch.autograd import Variable

from skimage.metrics import peak_signal_noise_ratio as Psnr#measure import compare_psnr as Psnr
from skimage.metrics import structural_similarity as ssim
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from misc import *
import models.mopnet as net
from models.vgg16 import Vgg16
from myutils import utils
import torch.nn.functional as F
import scipy.stats as st

import numpy as np
import cv2
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='exp', help='folder to output images and model checkpoints')
parser.add_argument('--dataset', required=False,
  default='my_loader',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--netCcol', help="path to classifier color network")
parser.add_argument('--netCgeo', help="path to classifier geo network")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netE', default="EdgePredictWeight/netG_epoch_33.pth", help="path to netE (to continue training)")
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=532, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=512, help='the height / width of the cropped input image to network')
parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--write', type=int, default=1, help='if write the results?')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0")

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

val_dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='test',
                       shuffle=False,
                       seed=opt.manualSeed,
                       pre=opt.pre)

inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# create directory to store test results
image_path=os.path.join(opt.exp_name,'test')
if os.path.exists(image_path):
  response=input('test directory already exists,,,Overwrite? [y/n]')
  if response=='y':
    os.remove(image_path)
  else:
    raise FileExistsError()
os.mkdir(image_path)
os.makedirs([os.path.join(image_path,sub) for sub in ['d','o','g']])

# Define the models
netG=net.Single()
netG.load_state_dict(torch.load(opt.netG))
netG.eval()
netG.to(device)
netEdge = net.EdgePredict()
netEdge.load_state_dict(torch.load(opt.netE))
netEdge.eval()
netEdge.to(device)
print(netG)

target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
target, input = target.to(device), input.to(device)

# Classifiers
net_label_color=net.vgg19ca()
net_label_color.load_state_dict(torch.load(opt.netCcol))
net_label_color=net_label_color.to(device)

net_label_geo = net.vgg19ca_2()
net_label_geo.load_state_dict(torch.load(opt.netCgeo))
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

vpsnr=0
vssim=0
for i, data in enumerate(val_dataloader, 0):

    input_cpu, target_cpu = data
    batch_size = target_cpu.size(0)

    # get paired data
    target_cpu, input_cpu = target_cpu.float().to(device), input_cpu.float().to(device)
    with torch.no_grad():
      target.resize_as_(target_cpu).copy_(target_cpu)
      input.resize_as_(input_cpu).copy_(input_cpu)

      i_G_x = conv1(input)
      i_G_y = conv2(input)
      iG = torch.tanh(torch.abs(i_G_x)+torch.abs(i_G_y))

      # predict color labels
      _, label_color = torch.max(net_label_color(input), 1)
      label_curve, label_thick = net_label_geo(iG)
      _, label_curve = torch.max(label_curve, 1)
      _, label_thick = torch.max(label_thick, 1)
      label_curve = label_curve.float()
      label_color = label_color.float()
      label_thick = label_thick.float()
      labels = [label_curve, label_color, label_thick]
      
      # Get input edges
      i_G_x_ = conv1(input)
      i_G_y_ = conv2(input)
      input_edge = torch.tanh(torch.abs(i_G_x_)+torch.abs(i_G_y_))

      # Get predicted edges
      edge1 = netEdge(torch.cat([input, input_edge], 1))
      _, edge = edge1

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
          # import pdb; pdb.set_trace()
          mi1 = cv2.cvtColor(utils.my_tensor2im(ti1), cv2.COLOR_BGR2RGB)
          mt1 = cv2.cvtColor(utils.my_tensor2im(tt1), cv2.COLOR_BGR2RGB)
          ori = cv2.cvtColor(utils.my_tensor2im(ori), cv2.COLOR_BGR2RGB)
          vpsnr+=Psnr(mi1,mt1)
          vssim+=ssim(mi1,mt1,multichannel=True)
          

          if opt.write==1:
              cv2.imwrite(os.path.join(image_path,'d','d'+str(i)+'_'+str(j) +'_.png'), mi1)
              cv2.imwrite(os.path.join(image_path,'o','o' + str(i)+'_'+str(j) + "_.png"), ori)
              cv2.imwrite(os.path.join(image_path, 'g','g' + str(i) + '_' + str(j) + "_.png"), mt1)
          import pdb; pdb.set_trace()
              
    
    print(50*'-')
    print(vcnt)
    print(50*'-')

avr_psnr = float(vpsnr)/vcnt
avr_ssim = float(vssim)/vcnt
stats_path = os.path.join(opt.exp,'test','test.txt')
stats=open(os.path.join(opt.exp,'test','test.txt'),'w')
stats.write('average psnr : {:>10f}, average ssim : {:>10f} '.format(avr_psnr, avr_ssim))
print('average psnr : {:>10f}, average ssim : {:>10f} '.format(avr_psnr, avr_ssim))