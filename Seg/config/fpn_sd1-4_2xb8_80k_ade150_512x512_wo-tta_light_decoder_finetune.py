_base_ = [
    '../_base_/models/fpn_sd.py','../_base_/datasets/ade150_sd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model_wrapper_cfg = dict(
                find_unused_parameters=True,
            )
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='DiffusionSeg',
    decode_head=dict(
        type='LightTextFPNHead',
        num_classes=150,
        in_channels=[256, 256, 256, 256],
        global_text_dim=768,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLossWithTemp', use_sigmoid=False, loss_weight=1.0, temperature=0.02)),
    backbone=dict(
        type='unet',
        version='1-4',
        dtype='float32',
        model_path='/data/sydong/diffusion/stable-diffusion-v1-5',
        img_size='512',
        max_attn_size=42,
        fronzen=False,
        cross_attn=True,
        place_in_unet=['mid', 'up'],
        use_checkpoint=True
    ),
    fuse=dict(in_dims=[1280, 1357, 1357, 960], arch='tiny', target='FPNHead',do_fuse=True),
    train_cfg=dict(
        meta_file='Seg/data/ade150.json',
    ),
    time_steps=[10],
    expert='resnet',
    second_last_layer=True,
)



train_dataloader = dict(batch_size=1, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)