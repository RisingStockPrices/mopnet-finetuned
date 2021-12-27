#####################################
# SET INFERENCE PARAMS IN common.sh #
#####################################
source common.sh

CUDA_VISIBLE_DEVICE=0 python scripts/inference.py \
    --dataroot $inference_dataroot \
    --exp $inference_dir \
    --netG $netG \
    --netE $netE \
    --netCcol $CLASSIFIER_PRETRAINED_COLOR \
    --netCgeo $CLASSIFIER_PRETRAINED_GEO \
    --imgW $imgW \
    --imgH $imgH \
    --imageCropSize $imageSize