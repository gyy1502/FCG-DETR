dataset_type = 'DOTADataset'
data_root = '/workspace/DATASET/DOTA_split/split_ss_dota1_0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25], ##修改已保持不变
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1, ####batch'si'ze
    workers_per_gpu=1,
    train=dict(
        type='DOTADataset',
        ann_file=
        '/workspace/DATASET/DOTA_split/split_ss_dota1_0/trainval_plane/annfiles/',
        img_prefix=
        '/workspace/DATASET/DOTA_split/split_ss_dota1_0/trainval_plane/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(1024, 1024)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.00, 0.00, 0.00],#####关掉反转以对齐可视化
                direction=['horizontal', 'vertical', 'diagonal'],
                version='le90'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
                 meta_keys=['filename', 'ori_filename', 'img_shape']) # 放进 meta)###实际控制
            ], 
            version='le90'),
    val=dict(
        type='DOTADataset',
        ann_file='/workspace/DATASET/DOTA_split/split_ss_dota1_0/tmp_test_big/annfiles/',
        img_prefix='/workspace/DATASET/DOTA_split/split_ss_dota1_0/tmp_test_big/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'),
    test=dict(
        type='DOTADataset',
        ann_file='/workspace/DATASET/DOTA_split/split_ss_dota1_0/test_beau/images/',  #refine_test
        img_prefix=
        '/workspace/DATASET/DOTA_split/split_ss_dota1_0/test_beau/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='le90'))
evaluation = dict(interval=1, metric='mAP', save_best='auto')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=1e-05,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[32])
runner = dict(type='EpochBasedRunner', max_epochs=64)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', interval=10, by_epoch=True)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
angle_version = 'le90' ##### ARS 的角度方式
model = dict(
    type='FCGDETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='./pretrain/resnet50-0676ba61.pth')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DNARSDeformableDETRHead',
        num_query=300,
        num_classes=15,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        angle_coder=dict(
            type='CSLCoder',
            angle_version='le90',
            omega=1,
            window='aspect_ratio',
            radius=6,
            normalize=True),
        focus_criterion=dict(
            type='HFGCriterion',
            noise_scale=0.0,
            alpha=0.25,
            gamma=2.0,
            save_gauss_results=dict(       
                label_nds_dir ='visual_res/label_nds/',
                ),
            weight=1.0),
        transformer=dict(
            type='DNARSRotatedDeformableDetrTransformer',
            two_stage_num_proposals=300,
            save_intermediate_results=dict(          
                heatmap_dir='visual_res/teaser/scoremaps_pixel/',  # Directory to save heatmaps
                ),
            encoder=dict(
                type='FCGTransformerEncoder',
                num_layers=6,
                encoder_layer=dict(
                    type='FCGTransformerEncoderLayer',
                    embed_dims=256,
                    n_heads=8,
                    dropout=0.0,
                    n_levels=4,
                    n_points=4,
                    d_ffn=2048)),        
                    # attn_cfgs=dict(
                    #     type='MultiScaleDeformableAttention', embed_dims=256),
                    # feedforward_channels=1024,
                    # ffn_dropout=0.1,
                    # operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DNARSDeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='RotatedMultiScaleDeformableAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='le90',
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1, 1, 1, 1, 1)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=2.0),
        loss_iou=dict(type='GIoULoss', loss_weight=5.0),
        reg_decoded_bbox=True,
        loss_angle=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0),
        rotate_deform_attn=True,
        aspect_ratio_weighting=True,
        dn_cfg=dict(
            type='DnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4, angle=0.02),
            group_cfg=dict(dynamic=True, num_groups=None,
                           num_dn_queries=100))),
    train_cfg=dict(
        assigner=dict(
            type='ARS_HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=2.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=5.0),
            angle_cost=dict(type='CrossEntropyLossCost', weight=2.0))),
    test_cfg=dict())
find_unused_parameters = True
work_dir = 'work_dirs/new_tmptrain/'
auto_resume = False
gpu_ids = range(0, 2)
