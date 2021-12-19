########################
# common params
dataroot=./aligned.v2
exp_name=fine-tuning

########################
# params for training 
display_port=8098 
batchSize=2
imageSize=256 # applies to test too

#########################
# params for inference
inference_dataroot=./aligned.v2/test/source
imgW=1920
imgH=1080


######################################
# pretrained weights for fine-tuning
CLASSIFIER_PRETRAINED_COLOR=./classifier/color_epoch_95.pth
CLASSIFIER_PRETRAINED_GEO=./classifier/geo_epoch_95.pth
EDGE_PRETRAINED=./mopnet/netEdge_epoch_150.pth
MAIN_PRETRAINED=./mopnet/netG_epoch_150.pth