# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Sequence
import copy
import numpy as np
from torch import Tensor, nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, xavier_init)
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)

from torch.nn.init import normal_

from .builder import ROTATED_TRANSFORMER
from mmdet.models.utils import Transformer
from mmdet.models.utils.transformer import inverse_sigmoid
# from mmrotate.core import obb2poly, poly2obb
# # from mmrotate.core import obb2xyxy
# from mmdet.core import bbox_cxcywh_to_xyxy
from mmcv.utils import build_from_cfg
from mmrotate.models.utils.ms_deform_attn import MultiScaleDeformableAttention



class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""
    def __init__(self, num_embeddings: int = 50, num_pos_feats: int = 256):
        super().__init__()
        self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
        self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask: Tensor):
        h, w = mask.shape[-2:]
        i = torch.arange(w, device=mask.device)
        j = torch.arange(h, device=mask.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim= -1,
            ).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


@TRANSFORMER_LAYER_SEQUENCE.register_module()       
class FCGTransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            embed_dims=256,
            d_ffn=1024,
            dropout=0.1,
            n_heads=8,
            # activation=nn.ReLU(inplace=True),
            n_levels=4,
            n_points=4,
            # focus parameter
            topk_sa=300,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.topk_sa = topk_sa
        # pre attention, batch_first batch维度在前
        self.pre_attention = nn.MultiheadAttention(embed_dims, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dims)

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dims, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dims)

        # ffn
        self.linear1 = nn.Linear(embed_dims, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dims)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dims)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.pre_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.pre_attention.out_proj.weight)
        # initilize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
            self,
            query,
            query_pos,
            value,  # focus parameter
            reference_points,
            spatial_shapes,
            level_start_index,
            query_key_padding_mask=None,
            # focus parameter
            score_tgt=None,  #11109
            foreground_pre_layer=None,
    ):
        mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
        # 在选出前300个 index
        select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]  #300
        # [bs,300,256]
        select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dims)
        # 前300个query
        select_tgt = torch.gather(query, 1, select_tgt_index)
        # 同上
        select_pos = torch.gather(query_pos, 1, select_tgt_index)

        query_with_pos = key_with_pos = self.with_pos_embed(select_tgt, select_pos)
        # [bs,300,256]
        tgt2 = self.pre_attention(  ##300
            query_with_pos,
            key_with_pos,
            select_tgt,  # 上面两个加了post，tgt是没有加的，相当于value
        )[0]

        select_tgt = select_tgt + self.pre_dropout(tgt2)  #300

        select_tgt = self.pre_norm(select_tgt)
        query = query.scatter(1, select_tgt_index, select_tgt) # [2,11109,256]但已经将选择后的300个放进去了

        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )     #[2,11109,256]

        query = query + self.dropout1(src2)
        query = self.norm1(query)
        # ffn
        query = self.forward_ffn(query) #[2,11109,256]

        return query

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class FCGTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dims = encoder_layer.embed_dims

        # learnt background embed for prediction
        self.background_embedding = PositionEmbeddingLearned(200, num_pos_feats=self.embed_dims // 2)
        self.encoder_class_head = nn.Linear(self.embed_dims, 15) ####15类
        self.enhance_mcsp = self.encoder_class_head

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    # Deformable DETR 生成点位
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [n, s, l, 2]
        return reference_points

    def forward(
            self,
            query,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_pos=None,
            query_key_padding_mask=None,
            foreground_score=None,
            focus_token_nums=None,  
            foreground_inds=None,  
            multi_level_masks=None,
    ):
        # 生成grid点位
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        b, n, s, p = reference_points.shape #[2,21760,4,2]
        ori_reference_points = reference_points  
        ori_pos = query_pos  #[2,21760,256]
        value = output = query
        for layer_id, layer in enumerate(self.layers):
            # [bs,s,256]
            # print("**************************************************************************layer_id",layer_id)
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dims)
            query = torch.gather(output, 1, inds_for_query)
            # 同上
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            # 同上
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)  
            ).view(b, -1, s, p)
            score_tgt = self.enhance_mcsp(query)   #91类

            query = layer(
                query,  
                query_pos,  
                value,  
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
                score_tgt,
                foreground_pre_layer,
            )
            outputs = []

            for i in range(foreground_inds[layer_id].shape[0]):
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                # 对应的query
                query_no_pad = query[i][:focus_token_nums[i]]
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),  # (x,) -> (x,256)
                        query_no_pad,
                    )
                )
            output = torch.stack(outputs)
        # add learnt embedding for background
        if multi_level_masks is not None:
            background_embedding = [
                self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
            ]
            # [bs,s,256]
            background_embedding = torch.cat(background_embedding, dim=1)
            background_embedding.scatter_(1, inds_for_query, 0)
            background_embedding *= (~query_key_padding_mask).unsqueeze(-1)

            output = output + background_embedding   #[2,21760,256] 背景嵌入相加

        return output
