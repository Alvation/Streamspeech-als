# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import torch

from fairseq import checkpoint_utils
from fairseq.models import register_model, register_model_architecture
# from fairseq.models.speech_to_speech.s2s_transformer import S2UTTransformerModel
from fairseq_modules.models.speech_to_speech.s2s_transformer import S2UTTransformerModel
from uni_unity.models.s2t_conformer import UniS2TConformerEncoder
from fairseq.models.transformer import Linear

logger = logging.getLogger(__name__)


def build_s2s_uni_conformer_encoder(args):
    encoder = UniS2SConformerEncoder(args)
    pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
    if pretraining_path is not None:
        if not Path(pretraining_path).exists():
            logger.warning(
                f"skipped pretraining because {pretraining_path} does not exist"
            )
        else:
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=pretraining_path
            )
            logger.info(f"loaded pretrained encoder from: {pretraining_path}")
    return encoder


class UniS2SConformerEncoder(UniS2TConformerEncoder):
    """Based on S2T transformer encoder, with support
    to incorporate target speaker embedding."""

    def __init__(self, args):
        super().__init__(args)

        self.spk_emb_proj = None
        if args.target_speaker_embed:
            self.spk_emb_proj = Linear(
                args.encoder_embed_dim + args.speaker_embed_dim, args.encoder_embed_dim
            )

    def forward(
        self, src_tokens, src_lengths, tgt_speaker=None, return_all_hiddens=False
    ):
        out = super().forward(src_tokens, src_lengths, return_all_hiddens)

        if self.spk_emb_proj:
            x = out["encoder_out"][0]
            seq_len, bsz, _ = x.size()
            tgt_speaker_emb = tgt_speaker.view(1, bsz, -1).expand(seq_len, bsz, -1)
            x = self.spk_emb_proj(torch.cat([x, tgt_speaker_emb], dim=2))
            out["encoder_out"][0] = x

        return out


class UniS2UTConformerModel(S2UTTransformerModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + Transformer discrete unit decoder
    """

    @staticmethod
    def add_args(parser):
        S2UTTransformerModel.add_args(parser)
        parser.add_argument(
            "--depthwise-conv-kernel-size",
            type=int,
            metavar="N",
            help="kernel size of depthwise convolution layers",
        )
        parser.add_argument(
            "--attn-type",
            type=str,
            metavar="STR",
            help="If not specified uses fairseq MHA. Other valid option is espnet for using conformer",
        )
        parser.add_argument(
            "--pos-enc-type",
            type=str,
            metavar="STR",
            help="Must be specified in addition to attn-type=espnet for rel_pos and rope",
        )

    @classmethod
    def build_encoder(cls, args):
        return build_s2s_uni_conformer_encoder(args)
