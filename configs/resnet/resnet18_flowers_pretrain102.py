_base_ = [
    '../_base_/models/resnet18_flowers.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_bs32_pretrain102.py',
    '../_base_/default_runtime.py'
]

# Path to the pre-trained weights learned on Oxford102
load_from = '/userhome/cs2/u3577254/mmpretrain/pretrain_pth_file/102_epoch_100.pth'

# Model configuration adjustments for fine-tuning
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),  # Only output the last stage feature map
        style='pytorch',
        frozen_stages=3,  # Freeze the first 3 stages (layers)
        init_cfg=dict(
            type='Pretrained',
            checkpoint=load_from,
            prefix='backbone',
        ),
    ),
    head=dict(
        type='LinearClsHead',
        num_classes=17,  # Number of classes for the new task
        in_channels=512,  # Number of input channels from ResNet18's layer4
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)