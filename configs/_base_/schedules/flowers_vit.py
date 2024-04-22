# I use AdamW, small init lr, warmup,several data augmentaion, and early stop if overfit may happen to avoid overfitting

optim_wrapper = dict(
    optimizer=dict(
        #use AdamW to avoid overfit
        type='AdamW',
        lr=5e-4 * 1024 / 512,  #adjust + small init
        weight_decay=1e-3,  # weight decay
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,  
        by_epoch=True,
        begin=20,
        end=100 
    )
]

train_cfg = dict(
    by_epoch=True,
    max_epochs=200,  #train 200 epochs instead of train 100 but with slow learning policy，the past usually get overfit and not stable
    val_interval=1
)

val_cfg = dict()
test_cfg = dict()

# add data agumentation
data_augmentation = dict(
    transforms=[
        dict(type='RandomResizedCrop', size=224),
        dict(type='RandomHorizontalFlip'),
        dict(type='ColorJitter'),
        #adding value of alpha considering “flowers” data
        #dict(type='Mixup', alpha=0.2),  
        #dict(type='CutMix', alpha=0.2),  
        #dict(type='AutoAugment'),  
        dict(type='RandomErasing', probability=0.1),  # 添加随机擦除，概率设为0.1
        #dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0),  # 添加高斯模糊
        dict(type='Perspective', distortion_scale=0.5, probability=0.5),  # 添加透视变换
        dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 添加标准化
    ]
)

#add early stopping to avoid overfitting.
early_stopping = dict(
    monitor='val_loss',
    patience=10,
    verbose=True,
    mode='min'
)