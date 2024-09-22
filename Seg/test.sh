PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python Seg/test.py --config Seg/configs/light_decoder_head/fpn_sd1-4_2xb8_80k_ade150_512x512_wo-tta_light_decoder_finetune.py \
                   --checkpoint experiments/Seg/ADE20K/diffusion/finetune/wo_neck_light_decoder/iter_80000.pth \
                   --work-dir experiments/Seg/COCO-stuff164k/diffusion/test/ \