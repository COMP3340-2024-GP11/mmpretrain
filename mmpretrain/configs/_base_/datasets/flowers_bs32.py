data = dict(
       samples_per_gpu=32,
       workers_per_gpu=1,
       train=dict(
           type=dataset_type,
           data_prefix='data/flowers/train',
           ann_file='data/flowers/meta/train.txt',
           pipeline=train_pipeline),
       val=dict(
           type=dataset_type,
           data_prefix='data/flowers/val',
           ann_file='data/flowers/meta/val.txt',
           pipeline=test_pipeline),
       test=dict(
           # replace `data/val` with `data/test` for standard test
           type=dataset_type,
           data_prefix='data/flowers/val',
           ann_file='data/flowers/meta/val.txt',
           pipeline=test_pipeline))
       evaluation = dict(interval=1, metric='accuracy')