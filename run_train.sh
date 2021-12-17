source common.sh

python train.py 
    --dataroot $data_root/train \
    --valDataroot $data_root/val \
    --exp $exp_name \
    --display_port $display_port \
    --imageSize $imageSize \
    --batchSize $batchSize \
    --netG $MAIN_PRETRAINED \
    --netE $EDGE_PRETRAINED \
    --netCcol $CLASSIFIER_PRETRAINED_COLOR \
    --netCgeo $CLASSIFIER_PRETRAINED_GEO \