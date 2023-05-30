_base_ = './fcosf_r50.py'

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
