_base_ = [
    '../_base_/models/vit_base_flowers.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_lrlow_pretrain_vit.py',
    '../_base_/tinyvit_save_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        frozen_stages = 6,
        type='VisionTransformer',
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            #I have put required pth file in sperate file due to it is very huge
            #change it to the exact path of vision_transformer/vit-base-p16_in21k.pth (the para of pretrian for vit)
            checkpoint='/userhome/cs2/u3577254/mmpretrain/pretrain_pth_file/vit-base-p16_in21k.pth',
            prefix='backbone'
        )
    ),
)
