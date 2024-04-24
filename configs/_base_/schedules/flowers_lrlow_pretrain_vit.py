optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.00005,  # freeze6 0.00005 freeze12 0.0001(not that intuitive to use a large lr for a linear layer learning anyway but indeed better than 0.00005)
        # freeze0 0.0001
        weight_decay=0.01,  
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=True,
        end=20,  # I let warmup decrease from 40 to 20 considring overall smooth training
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=60,  # cosine annealing持续40个epochs
        eta_min=1e-8,
        by_epoch=True,
        begin=20,  # 从第20个epoch开始
        end=80),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=0.05,  # 使用较低的常量学习率0.05
        begin=80,
        end=100)
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=5)
val_cfg = dict()
test_cfg = dict()
