_base_ = [
    '../_base_/models/seresnet50_flowers.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_bs32se.py',
    '../_base_/default_runtime.py'
]