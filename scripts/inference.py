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
import numpy as np
import cv2

from PIL import Image
import torchvision.transforms as transforms

torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

## manually set arg default below ##
parser.add_argument('--netG', default='checkpoints/netG_epoch_58.pth', help="path to main network checkpoint")
parser.add_argument('--netE', default="checkpoints/netEdge_epoch_58.pth", help="path to edge network checkpoint")
parser.add_argument('--imgW', type=int, default=1080)
parser.add_argument('--imgH', type=int, default=720)
#############################

parser.add_argument('--dataroot', default='./aligned', help='path to images for demoire-ing')
parser.add_argument('--exp', default='inference', help='name of directory to be created to store demoire results')
parser.add_argument('--netCcol', default='./classifier/color_epoch_95.pth',help="path to classifier color network")
parser.add_argument('--netCgeo', default='./classifier/geo_epoch_95.pth',help="path to classifier geo network")
parser.add_argument('--dataset', required=False,
  default='my_loader',  help='')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=532, help='the height / width of the original input image')
parser.add_argument('--imageCropSize', type=int, default=256)
parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--number', type=int, default=10)
opt = parser.parse_args()
print(opt)


device = torch.device("cuda:0")

opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# create directory to store test results
image_path=opt.exp
if os.path.exists(image_path):
  response=input('inference directory {} already exists,,,Overwrite? [y/n]'.format(image_path))
  if response=='y':
    import shutil
    shutil.rmtree(image_path)
  else:
    raise FileExistsError()
os.mkdir(image_path)
utils.mkdirs([os.path.join(image_path,sub) for sub in ['results','original']])


netG=net.Single()
netG.load_state_dict(torch.load(opt.netG))
netG.eval()
netG.to(device)
netEdge = net.EdgePredict()
netEdge.load_state_dict(torch.load(opt.netE))
netEdge.eval()
netEdge.to(device)
print(netG)

# target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imgH, opt.imgH)
# input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imgH, opt.imgW)
# target, input = target.to(device), input.to(device)

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

import os

mean = ((0.5,0.5,0.5))
std = ((0.5,0.5,0.5))
transform = transforms.Compose([ 
  transforms.Resize((opt.imgH,opt.imgW)),
  transforms.ToTensor(),
  transforms.Normalize(mean, std),])

transform_classifier = transforms.Compose([ 
  transforms.CenterCrop(opt.imageCropSize),
  transforms.ToTensor(),
  transforms.Normalize(mean, std),])

vcnt=0
for file in os.listdir(opt.dataroot):
  
  path = os.path.join(opt.dataroot,file)
  img = Image.open(path).convert('RGB')

  input = transform(img)
  input = input.float().to(device)
  input_cropped=transform_classifier(img)
  input_cropped = input_cropped.float().to(device)
  
  with torch.no_grad():
    input.resize_as_(input).copy_(input)
    input = input.unsqueeze(0)
    input_cropped = input_cropped.unsqueeze(0)

    i_G_x = conv1(input_cropped)
    i_G_y = conv2(input_cropped)
    iG = torch.tanh(torch.abs(i_G_x)+torch.abs(i_G_y))

    # import pdb; pdb.set_trace()
    # predict color labels
    _, label_color = torch.max(net_label_color(input_cropped), 1)
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
    input.cuda()
    edge.cuda()
    
    # Moire removal
    x_hat1 = netG(input, edge, labels)
    residual, x_hat = x_hat1

    # Save results
    for j in range(x_hat.shape[0]):
        vcnt += 1
        b, c, w, h = x_hat.shape
        ti1 = x_hat[j, :,:,: ]
        ori = input[j, :, :, :]
        
        mi1 = cv2.cvtColor(utils.my_tensor2im(ti1), cv2.COLOR_BGR2RGB)
        ori = cv2.cvtColor(utils.my_tensor2im(ori), cv2.COLOR_BGR2RGB)
        
        cv2.imwrite(image_path + os.sep+'results'+os.sep+file, mi1)
        cv2.imwrite(image_path + os.sep+'original'+os.sep+file,ori)
                
    print(50*'-')
    print(vcnt)
    print(50*'-')

print(50*'-')