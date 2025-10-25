# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
import math, copy
from typing import Any, Optional, Tuple, Type
import torch.nn.functional as F

try:
    from .common import LayerNorm2d
except:
    from common import LayerNorm2d

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Semantic_MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(Semantic_MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        cls_n = value.shape[-1]
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]
        value = value.repeat(self.h, 1, 1, 1).view(self.h, nbatches, -1, cls_n).permute(1,0, 2, 3).contiguous()

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # attn_res_list = []
        # for i in range(cls_n):
        #     attn_res, self.attn = attention(query, key, value[..., i:i+1], mask=mask, dropout=self.dropout, attn_score=self.attn)
        #
        #     attn_res_list.append(attn_res)
        #
        # attn_res = torch.cat(attn_res_list,dim=-1).to(value)

        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)

def attention(query, key, value, mask=None, dropout=None, attn_score=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    if attn_score is None:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
    else:
        p_attn = attn_score
        # if dropout is not None:
        #     p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn




class AutoSeAMv3_PromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            image_embedding_size: Tuple[int, int],
            input_image_size: Tuple[int, int],
            mask_in_chans: int,
            class_num: int,
            activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input mask
          class_num: the number of semantic class
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.class_num = class_num
        # 定义learnable token，包括前景和背景点（AutoSeAM没用，可以考虑做成每个类别的pos/neg point），每个类别的bbox（两点坐标）
        self.num_point_embeddings: int = 2 + (2 * class_num)  # pos/neg point + 2*class_num box corners with class
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.not_a_box_point1_embed = nn.Embedding(1, embed_dim)
        self.not_a_box_point2_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.sematic_mhsa = Semantic_MultiHeadedAttention(h=8, d_model=embed_dim, dropout=.5)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(class_num + 1, mask_in_chans // 4, kernel_size=2, stride=2),  #  stride=1,padding=1
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),  #  stride=1,padding=1
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )  # downsample to 1/4
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
            self,
            points: torch.Tensor,
            labels: torch.Tensor,
            pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts.添加类别语义嵌入"""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _embed_boxes_label_embed(self, boxes: torch.Tensor, box_labels: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts.添加类别语义嵌入"""
        bs = boxes.shape[0]
        box_labels = torch.repeat_interleave(box_labels, 2, dim=1)
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(bs, -1, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        # 初始化没有label的box
        corner_embedding[box_labels == -1] = 0.0
        corner_embedding[box_labels == -1][0::2, :] += self.not_a_box_point1_embed.weight
        corner_embedding[box_labels == -1][1::2, :] += self.not_a_box_point2_embed.weight
        # 初始化每个lable的box, class的index从1开始
        for cls in range(0, self.class_num):
            corner_embedding[(box_labels - 1) == cls][0::2, :] += self.point_embeddings[2 * cls + 2].weight
            corner_embedding[(box_labels - 1) == cls][1::2, :] += self.point_embeddings[2 * cls + 3].weight

        # corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        # corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks_label_embed(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs. 添加类别语义嵌入"""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes['bbox'].shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[Tuple[torch.Tensor, torch.Tensor]],
            masks: Optional[torch.Tensor],
            query_feat,
            support_feat,
            dense_prompt_type='conv_add',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            boxes, box_labels = boxes['bbox'], boxes['bbox_label']
            box_embeddings = self._embed_boxes_label_embed(boxes, box_labels)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            # dense_embeddings = self._embed_masks_label_embed(masks)
            # support_masks [B, Shot, K, H, W]
            # support_feats [B, Shot, C, h, w]
            # query_feat [B, C, h, w]

            if dense_prompt_type == 'conv_add':
                bsz, ch, ha, wa = query_feat.shape
                mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                masks = torch.stack(
                    [F.interpolate(sm.contiguous().view(-1, mask_h, mask_w).unsqueeze(1).float(), (4*ha, 4*wa),
                                   mode='bilinear', align_corners=True).contiguous().view(self.class_num + 1, 4*ha, 4*wa)
                     for sm in masks])
                dense_embeddings = self._embed_masks_label_embed(masks)
            else:
                bsz, ch, ha, wa = query_feat.shape
                mask_h, mask_w = masks.shape[-2], masks.shape[-1]
                support_num = support_feat.shape[1]
                assert masks.shape[2] == self.class_num + 1
                query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
                support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                support_mask = masks.contiguous()
                mask = torch.stack([F.interpolate(sm.contiguous().view(-1, mask_h, mask_w).unsqueeze(1).float(), (ha, wa),
                                                  mode='bilinear', align_corners=True).view(support_num, self.class_num + 1, ha, wa)
                                    for sm in support_mask])

                mask = mask.view(-1, self.class_num + 1, ha * wa).permute(0, 2, 1).contiguous()

                coarse_mask = self.sematic_mhsa(query, support_feat, mask)  # query:[B, N, C], support_feat:[B*K, N, C], mask:[B, K, CLS H, W]
                coarse_mask = coarse_mask.view(bsz, ha, wa, self.class_num + 1).permute(0, 3, 1, 2).contiguous()

                dense_embeddings = self._embed_masks_label_embed(coarse_mask)

        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


if __name__ == '__main__':
    prompt_encoder = AutoSeAMv3_PromptEncoder(embed_dim=128, image_embedding_size=(32, 32), input_image_size=(512, 512),
                                            mask_in_chans=16, class_num=12)
    box_labels = -1*torch.ones((2, 5))
    box_labels[0, 0] = 1
    box_labels[0, 4] = 4
    box_labels[0, 2] = 12
    # box_labels = torch.repeat_interleave(box_labels,2,dim=1)AutoSeAM_PromptEncoder, AutoSeAMv2_PromptEncoder, AutoSeAM_MaskDecoder, AutoSeAMv2_MaskDecoder
    box = (torch.randn((2, 5, 4)), box_labels)
    points = (torch.randn((2, 10, 2)), torch.zeros((2, 10)))
    masks = None
    points = None
    box = None
    mask = torch.randn((2, 1, 13, 512, 512))
    query = torch.randn((2, 128, 32, 32))
    key = torch.randn((2, 1, 128, 32, 32))
    sparse_embeddings, dense_embeddings = prompt_encoder(points, box, mask, query, key, 'conv_add')
    print(sparse_embeddings.shape, dense_embeddings.shape)


