######################################
data_root=/aligned.v2

exp_name=fine-tuning

######################################
# pretrained weights for fine-tuning
CLASSIFIER_PRETRAINED_COLOR=./color_epoch_95.pth
CLASSIFIER_PRETRAINED_GEO=./geo_epoch_95.pth

EDGE_PRETRAINED=./mopnet/netEdge_epoch_150.pth
MAIN_PRETRAINED=./mopnet/netG_epoch_150.pth

########################
# params for training
display_port=8098 
batchSize=2
imageSize=256
exp_name=fine-tuning