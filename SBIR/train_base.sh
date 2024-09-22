
python SBIR/train_SBIR_BLIP_train_test.py\
    --cuda 1 \
    --fuse_arch add \
    --expert no_expert \
    --seed 1 \
    --commit wo_fuse_wo_epxert_add \
    --max_attn_size 16 \
    --content main/BLIP_train_test \
    --time_step 200 \
    --cross_attn yes \
    --do_fuse no \
    --inv_method ddim_inv \
    --place_in_unet mid up \








