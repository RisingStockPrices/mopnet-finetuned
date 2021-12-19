#####################################
# SET INFERENCE PARAMS IN common.sh #
#####################################
source common.sh

exp_dir=$exp_name/checkpoints
netG=$(find ${exp_dir} --name netG_epoch_*.pth | sort | tail -n 1)
netE=$(find ${exp_dir} --name netEdge_epoch_*.pth | sort | tail -n 1)

CUDA_VISIBLE_DEVICE=0 python scripts/inference.py \
    --dataroot $inference_dataroot \
    --netG $netG \
    --netE $netE \
    --imgW $imgW \
    --imgH $imgH \
    --gt_provided \ # saves gt results as well if specified, usually would be set to false