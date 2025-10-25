# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
# from icecream import ic
from torch import nn
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, AutoSeAMv2_PromptEncoder, AutoSeAMv3_PromptEncoder, Sam, TwoWayTransformer,\
    Sam_two_Decoder, Sam25d, Sam25d_Box_Generator, AutoSam25d_Box_Generator, MaskDecoder_AutoSeAM, \
    AutoSeAMv2
from isegm.model.sam25d_module.Space_Slice_Attention import Space_Slice_Attention_Block
# from simpleclick.isegm.model.sam25d_module.Box_Generator import AnchorPromptEncoder, BoxGenerateDecoder, BoxGenerator, BoxGenerateTransformer
from isegm.model.sam25d_module.Box_Generator import *
from isegm.model.sam25d_module.AnchorQuery_DETR import AnchorQuery_Detector, AnchorQuery_Decoder
from isegm.model.sam25d_module.AnchorQuery_PE import PositionEmbeddingSine, ClassEmbedding
from .AutoSeAMv2_fact_tt_image_encoder import Fact_tt_AutoSeAMv2
# from isegm.model.segment_anything_scale.AutoSeAMv2_fact_tt_image_encoder import Fact_tt_AutoSeAMv2

def build_sam_vit_h(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


def build_sam_vit_b(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )

def build_sam25d_vit_b(image_size, out_num, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam25d(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        master_out_number=out_num,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        attention_type='divided_space_time'
    )

def build_sam25d_box_generator_vit_b(image_size, out_num, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None, attention_type='divided_space_time'):
    return _build_sam25d_box_generator(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        master_out_number=out_num,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        attention_type='divided_space_time'
    )

def build_AutoSam25d_box_generator_vit_b(image_size, out_num, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None, attention_type='divided_space_time'):
    return _build_AutoSam25d_box_generator(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        master_out_number=out_num,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        attention_type='divided_space_time'
    )

# 2023-11-1
def build_AutoSeAM_vit_b(image_size, out_num, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None, attention_type='divided_space_time'):
    return _build_AutoSeAM(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        master_out_number=out_num,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        attention_type='divided_space_time'
    )

def build_AutoSeAMv2_vit_b(image_size, out_num, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None, attention_type='divided_space_time'):
    return _build_AutoSeAMv2(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        master_out_number=out_num,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        attention_type='divided_space_time'
    )

def build_AutoSeAMv3_vit_b(image_size, out_num, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None, attention_type='divided_space_time'):
    return _build_AutoSeAMv3(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        master_out_number=out_num,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        attention_type='divided_space_time'
    )

def build_sam_2d_vit_b(image_size, master_out_num, second_out_num, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam_two_decoder(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        master_out_number=master_out_num,
        second_out_number=second_out_num,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        num_classes,
        image_size,
        pixel_mean,
        pixel_std,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size)
            sam.load_state_dict(new_state_dict)
    return sam, image_embedding_size

def _build_sam25d(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        master_out_number,
        image_size,
        pixel_mean,
        pixel_std,
        attention_type='divided_space_time',
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    master_sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=master_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    master_sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            master_sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(master_sam, state_dict, image_size, vit_patch_size)
            master_sam.load_state_dict(new_state_dict)
    ssa_depth = 2

    # # divided spatial slice
    # space_slice_attention_blocks = nn.ModuleList([Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=False) for i in range(ssa_depth)]
    #                                              + [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=True)])
    # spatial slice attention
    space_slice_attention_blocks = nn.ModuleList(
        [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=False, attention_type=attention_type) for i in range(ssa_depth)]
        + [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=True, attention_type=attention_type)])

    sam25d = Sam25d(master_sam, space_slice_attention_blocks)
    sam25d.train()

    return sam25d, image_embedding_size

def _build_sam25d_box_generator(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        master_out_number,
        image_size,
        pixel_mean,
        pixel_std,
        attention_type='divided_space_time',
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    master_sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=master_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    master_sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            master_sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(master_sam, state_dict, image_size, vit_patch_size)
            master_sam.load_state_dict(new_state_dict)
    ssa_depth = 2

    # spatial slice attention
    space_slice_attention_blocks = nn.ModuleList(
        [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=False, attention_type=attention_type) for i in range(ssa_depth)]
        + [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=True, attention_type=attention_type)])

    box_generator = BoxGenerator(
        anchor_prompt_encoder=AnchorPromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size)
        ),
        box_generator_decoder=BoxGenerateDecoder(
            transformer=BoxGenerateTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            head_depth=3,
            head_hidden_dim=256,
        )
    )

    sam25d_box_generator = Sam25d_Box_Generator(master_sam, space_slice_attention_blocks, box_generator)
    sam25d_box_generator.train()

    return sam25d_box_generator, image_embedding_size

# 2023-08-05
def _build_AutoSam25d_box_generator(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        master_out_number,
        image_size,
        pixel_mean,
        pixel_std,
        attention_type='divided_space_time',
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    master_sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=master_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    master_sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            master_sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(master_sam, state_dict, image_size, vit_patch_size)
            master_sam.load_state_dict(new_state_dict)
    ssa_depth = 2

    # spatial slice attention
    space_slice_attention_blocks = nn.ModuleList(
        [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=False, attention_type=attention_type) for i in range(ssa_depth)]
        + [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=True, attention_type=attention_type)])

    boxes_detect_decoder = AnchorQuery_Decoder(embed_dim=prompt_embed_dim,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=1024,  # 2048
            ffn_dropout=0.0,
            activation=nn.PReLU(),
            num_layers=6,
            modulate_hw_attn=True,)

    position_embedding = PositionEmbeddingSine(
        num_pos_feats=128,
        temperature=20,
        normalize=True,
    )

    box_generator = AnchorQuery_Detector(decoder=boxes_detect_decoder, position_embedding=position_embedding, embed_dim=prompt_embed_dim)

    sam25d_box_generator = AutoSam25d_Box_Generator(master_sam, space_slice_attention_blocks, box_generator)
    sam25d_box_generator.train()

    return sam25d_box_generator, image_embedding_size


def _build_AutoSeAM(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        master_out_number,
        image_size,
        pixel_mean,
        pixel_std,
        attention_type='divided_space_time',
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    master_sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder_AutoSeAM(
            # num_multimask_outputs=3,
            num_multimask_outputs=master_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    master_sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            master_sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(master_sam, state_dict, image_size, vit_patch_size)
            master_sam.load_state_dict(new_state_dict)
    ssa_depth = 2

    # spatial slice attention
    space_slice_attention_blocks = nn.ModuleList(
        [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=False, attention_type=attention_type) for i in range(ssa_depth)]
        + [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=True, attention_type=attention_type)])

    boxes_detect_decoder = AnchorQuery_Decoder(embed_dim=prompt_embed_dim,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=1024,  # 2048
            ffn_dropout=0.0,
            activation=nn.PReLU(),
            num_layers=6,
            modulate_hw_attn=True,)

    position_embedding = PositionEmbeddingSine(
        num_pos_feats=128,
        temperature=20,
        normalize=True,
    )

    class_embedding = ClassEmbedding(feat_dim=128)

    box_generator = AnchorQuery_Detector(decoder=boxes_detect_decoder, position_embedding=position_embedding, embed_dim=prompt_embed_dim)

    autoseamv2 = AutoSeAMv2(master_sam, space_slice_attention_blocks, box_generator)
    autoseamv2.train()

    return autoseamv2, image_embedding_size


def _build_AutoSeAMv2(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        master_out_number,
        image_size,
        pixel_mean,
        pixel_std,
        attention_type='divided_space_time',
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    master_sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=AutoSeAMv2_PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            class_num=master_out_number
        ),
        mask_decoder=MaskDecoder_AutoSeAM(
            # num_multimask_outputs=3,
            num_multimask_outputs=master_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    master_sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            master_sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(master_sam, state_dict, image_size, vit_patch_size)
            master_sam.load_state_dict(new_state_dict)
    ssa_depth = 2

    # spatial slice attention
    space_slice_attention_blocks = nn.ModuleList(
        [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=False, attention_type=attention_type) for i in range(ssa_depth)]
        + [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=True, attention_type=attention_type)])

    autoseamv2 = AutoSeAMv2(master_sam, space_slice_attention_blocks, 4)

    return autoseamv2, image_embedding_size

def _build_AutoSeAMv3(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        master_out_number,
        image_size,
        pixel_mean,
        pixel_std,
        attention_type='divided_space_time',
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    master_sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=AutoSeAMv3_PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            class_num=master_out_number
        ),
        mask_decoder=MaskDecoder_AutoSeAM(
            # num_multimask_outputs=3,
            num_multimask_outputs=master_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    master_sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            master_sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(master_sam, state_dict, image_size, vit_patch_size)
            master_sam.load_state_dict(new_state_dict)
    ssa_depth = 2

    # spatial slice attention
    space_slice_attention_blocks = nn.ModuleList(
        [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=False, attention_type=attention_type) for i in range(ssa_depth)]
        + [Space_Slice_Attention_Block(dim=prompt_embed_dim, num_heads=8, last_layer=True, attention_type=attention_type)])

    autoseamv2 = AutoSeAMv2(master_sam, space_slice_attention_blocks, 4)

    return autoseamv2, image_embedding_size

def _build_sam_two_decoder(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        master_out_number,
        second_out_number,
        image_size,
        pixel_mean,
        pixel_std,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
    master_sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=master_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # pixel_mean=[123.675, 116.28, 103.53],
        # pixel_std=[58.395, 57.12, 57.375],
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    # sam.eval()
    master_sam.train()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        try:
            master_sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from(master_sam, state_dict, image_size, vit_patch_size)
            master_sam.load_state_dict(new_state_dict)

    second_decoder = MaskDecoder(
            # num_multimask_outputs=3,
            num_multimask_outputs=second_out_number,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    sam = Sam_two_Decoder(master_sam, second_decoder)
    sam.train()


    return sam, image_embedding_size


def load_from(sam, state_dict, image_size, vit_patch_size):
    sam_dict = sam.state_dict()
    # 将改动过的参数和新添加的参数剔除
    except_keys = ['semantic_mhsa', 'mask_downscaling', 'mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[
                          2] not in k and except_keys[3] not in k and except_keys[
                          4] not in k}
    # except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    # new_state_dict = {k: v for k, v in state_dict.items() if
    #                   k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict
