source common.sh

python inference.py --dataroot "./data/source" --netG './checkpoints/netG_epoch_237.pth' --netE "./checkpoints/netEdge_epoch_237.pth" --image_path "results-237" 