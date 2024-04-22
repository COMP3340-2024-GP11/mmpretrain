optim_wrapper = dict(
    # optimizer serting
    optimizer=dict(
        type='AdamW',
        lr=0.0005,  # low lr (finetune)
        weight_decay=0.01,  # decrease the weight decay
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

# learning rate adjustment line
param_scheduler = [
    # warm up
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,  # approiate warm-up epochs
        convert_to_iter_based=True),
    # CosineAnnealingLR
    dict(type='CosineAnnealingLR', eta_min=1e-4, by_epoch=True, begin=20, end=100)  # 调整eta_min以保持一定的学习率
]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(base_batch_size=32)