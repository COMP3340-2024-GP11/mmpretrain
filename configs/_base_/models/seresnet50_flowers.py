# this is my model of se-resnet50
# SEResNet integrates the Squeeze-and-Excitation (SE) blocks into the ResNet architecture to enhance feature recalibration
# This is a explore of block attention.
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SEResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=17,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))