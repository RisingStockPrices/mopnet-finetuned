#################################
# SET TEST PARAMS IN common.sh  #
#################################
source common.sh

exp_dir=$exp_name/checkpoints
netG=$(find ${exp_dir} -name netG_epoch_*.pth | sort | tail -n 1)
netE=$(find ${exp_dir} -name netEdge_epoch_*.pth | sort | tail -n 1)

CUDA_VISIBLE_DEVICE=0 python scripts/test.py \
 --dataroot $dataroot/test \
 --exp=$exp_name \
 --netG $netG \
 --netE $netE \
 --netCcol $CLASSIFIER_PRETRAINED_COLOR \
 --netCgeo $CLASSIFIER_PRETRAINED_GEO \
 --batchSize 1 \
 --imageSize $imageSize \
 --write 1 \