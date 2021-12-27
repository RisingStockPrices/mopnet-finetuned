########################
# common params
dataroot=./aligned.v3
exp_name=fine-tuning-with-crop
description='added center cropping as original authors to relieve blurriness in training data'

########################
# params for training 
display_port=8098 
batchSize=2
imageSize=256 # applies to test too

#########################
# params for inference
inference_dataroot=./aligned.v3/test/source # image dir
imgW=1080 #1920
imgH=720 #1080
inference_dir=./inference #name of folder to save results
netG=./checkpoints/netG_epoch_58.pth
netE=./checkpoints/netEdge_epoch_58.pth

######################################
# pretrained weights for fine-tuning
CLASSIFIER_PRETRAINED_COLOR=./classifier/color_epoch_95.pth
CLASSIFIER_PRETRAINED_GEO=./classifier/geo_epoch_95.pth
EDGE_PRETRAINED=./mopnet/netEdge_epoch_150.pth
MAIN_PRETRAINED=./mopnet/netG_epoch_150.pth