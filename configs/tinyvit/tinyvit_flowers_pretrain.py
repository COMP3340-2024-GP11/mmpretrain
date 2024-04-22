_base_ = [
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_tinyvit_bs32_pretrain.py',
    '../_base_/default_runtime.py',
    '../_base_/models/tinyvit/tinyvit_flowers.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        frozen_stages = 3,
        init_cfg=dict(
            _delete_=True,
            type='Pretrained',
            #change it to the exact path of tinyvit-5m_in21k-distill-pre_3rdparty_in1k_20221021-d4b010a8.pth
            checkpoint='/userhome/cs2/u3577254/mmpretrain/pretrain_pth_file/tinyvit-5m_in21k-distill-pre_3rdparty_in1k_20221021-d4b010a8.pth',
            prefix='backbone'
        )
    ),
)