#################################
# SET TRAIN PARAMS IN common.sh #
#################################
source common.sh

# specify cuda device here
CUDA_VISIBLE_DEVICE=0 python scripts/train.py \
    --dataroot $dataroot/train \
    --valDataroot $dataroot/val \
    --exp $exp_name \
    --display_port $display_port \
    --imageSize $imageSize \
    --batchSize $batchSize \
    --netG $MAIN_PRETRAINED \
    --netE $EDGE_PRETRAINED \
    --netCcol $CLASSIFIER_PRETRAINED_COLOR \
    --netCgeo $CLASSIFIER_PRETRAINED_GEO \