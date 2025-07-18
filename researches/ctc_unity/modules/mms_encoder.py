import sys,logging

import torch
import torch.nn as nn
import math
import os
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, register_model
from fairseq.dataclass import FairseqDataclass
from argparse import Namespace
from transformers import WhisperForConditionalGeneration
from .sub_model.modules import WhisperEncoderWrapper, Projector, Multimodal_Attention, Speech_Rate_Predictor
from .sub_model.Qformer import BertConfig, BertLMHeadModel

logger = logging.getLogger(__name__)

@dataclass
class MultimodalEncoderConfig(FairseqDataclass):
    whisper_embed_dim: int = field(
        default=1024, metadata={"help": "whisper embedding dimension"}
    )
    avhubert_embed_dim: int = field(
        default=1024, metadata={"help": "avhubert embedding dimension"}
    )
    modality_fuse: str = field(
        default='concat', metadata={'help': 'fusing two modalities: concat, add, cross-att'}
    )
    use_qformer: bool = field(
        default=True, metadata={"help": "whether to use Q-Former for feature compression"}
    )
    queries_per_sec: int = field(
        default=4, metadata={"help": "queries per second for Q-Former"}
    )
    qformer_layers: int = field(
        default=2, metadata={"help": "number of Q-Former layers"}
    )
    qformer_dim: int = field(
        default=1024, metadata={"help": "Q-Former hidden dimension"}
    )
    use_sr_predictor: bool = field(
        default=False, metadata={"help": "whether to use speech rate predictor"}
    )
    sr_predictor_layers: int = field(
        default=2, metadata={"help": "number of speech rate predictor layers"}
    )

@register_model("multimodal_encoder", dataclass=MultimodalEncoderConfig)
class MultimodalEncoder(BaseFairseqModel):
    def __init__(self, avhubert, whisper, cfg):
        super().__init__()
        self.cfg = cfg
        self.avhubert = avhubert
        self.whisper = whisper
        
        # Freeze pretrained models
        for param in self.avhubert.parameters():
            param.requires_grad = False
        for param in self.whisper.parameters():
            param.requires_grad = False
        
        # Modality fusion
        self.modality_fuse = cfg.modality_fuse
        if self.modality_fuse == 'concat':
            self.embed_dim = cfg.whisper_embed_dim + cfg.avhubert_embed_dim
        elif self.modality_fuse == 'add':
            self.embed_dim = cfg.whisper_embed_dim
        elif self.modality_fuse == 'cross-att':
            self.multimodal_attention_layer = Multimodal_Attention(
                embed_dim=cfg.whisper_embed_dim,
                num_heads=8
            )
            self.embed_dim = cfg.whisper_embed_dim
        
        # Feature processing
        self.afeat_1d_conv = nn.Conv1d(
            in_channels=cfg.whisper_embed_dim,
            out_channels=cfg.whisper_embed_dim,
            kernel_size=2 if cfg.use_qformer else 4,
            stride=2 if cfg.use_qformer else 4,
            padding=0
        )
        
        # Q-Former
        if cfg.use_qformer:
            max_queries = int(cfg.queries_per_sec * 20 * (2 if cfg.use_sr_predictor else 1))
            qformer_config = BertConfig.from_pretrained("bert-large-uncased")
            qformer_config.num_hidden_layers = cfg.qformer_layers
            qformer_config.encoder_width = self.embed_dim
            qformer_config.hidden_size = cfg.qformer_dim
            qformer_config.add_cross_attention = True
            qformer_config.cross_attention_freq = 1
            qformer_config.query_length = max_queries
            
            self.Qformer = BertLMHeadModel(config=qformer_config)
            self.query_tokens = nn.Parameter(
                torch.zeros(1, max_queries, qformer_config.hidden_size)
            )
            self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
            
            if cfg.use_sr_predictor:
                self.sr_predictor = Speech_Rate_Predictor(num_layers=cfg.sr_predictor_layers)
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                sr_ckpt_path = f'{root_dir}/pretrained_models/sr_predictor/checkpoint.pt'
                sr_state = torch.load(sr_ckpt_path)['model']
                sr_state_ = {k[13:]: v for k, v in sr_state.items()}
                self.sr_predictor.load_state_dict(sr_state_)
                for param in self.sr_predictor.parameters():
                    param.requires_grad = False

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""
        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        w2v_path = f'{root_dir}/pretrained_models/avhubert/large_vox_iter5.pt'

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        w2v_args.task.data = cfg.data
        task_pretrain = tasks.setup_task(w2v_args.task)
        
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)
        avhubert = HubertEncoderWrapper(encoder_)
        
        if state is not None and not cfg.no_pretrained_weights:
            del state['model']['mask_emb']
            avhubert.w2v_model.load_state_dict(state["model"], strict=False)

        avhubert.w2v_model.remove_pretraining_modules()

        # Whisper initialization
        whisper_ = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium.en"
        ).model.encoder
        whisper = WhisperEncoderWrapper(whisper_)

        return cls(avhubert, whisper, cfg)

    def forward(self, source, padding_mask=None):
        """
        Args:
            source: dict containing 'audio' and/or 'video' tensors
            padding_mask: optional padding mask for AV-HuBERT
        Returns:
            encoded features (B x T x D)
        """
        # Extract features
        with torch.no_grad():
            # Whisper features (audio only)
            whisper_feats = self.whisper(source['audio']) if 'audio' in source else None
            
            # AV-HuBERT features (video only)
            avhubert_input = {'audio': None, 'video': source.get('video')}
            avhubert_out = self.avhubert(
                source=avhubert_input,
                padding_mask=padding_mask
            )
            avhubert_feats = avhubert_out['encoder_out'].transpose(0, 1)  # T x B x D -> B x T x D
        
        # Process Whisper features
        if whisper_feats is not None:
            whisper_feats = self.afeat_1d_conv(whisper_feats.transpose(1, 2)).transpose(1, 2)
            # Align temporal dimensions
            T = min(whisper_feats.size(1), avhubert_feats.size(1))
            whisper_feats = whisper_feats[:, :T, :]
            avhubert_feats = avhubert_feats[:, :T, :]
            
            # Fuse modalities
            if self.modality_fuse == 'concat':
                features = torch.cat([whisper_feats, avhubert_feats], dim=-1)
            elif self.modality_fuse == 'add':
                features = whisper_feats + avhubert_feats
            elif self.modality_fuse == 'cross-att':
                features = self.multimodal_attention_layer(
                    audio_feature=whisper_feats,
                    visual_feature=avhubert_feats
                )
        else:
            features = avhubert_feats
        
        # Optional Q-Former compression
        if self.cfg.use_qformer:
            features = self.compress_with_qformer(features, padding_mask)
            
        return features

    def compress_with_qformer(self, features, padding_mask=None):
        """Compress features using Q-Former"""
        B, T, _ = features.size()
        
        # Calculate query lengths
        if self.cfg.use_sr_predictor:
            len_queries, resized_len_list = self.calculate_query_lengths(features)
        else:
            len_queries = [max(int(T / 25 * self.cfg.queries_per_sec), 1)] * B
        
        max_queries = max(len_queries)
        
        # Prepare query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)[:, :max_queries, :]
        query_attn_mask = torch.zeros(B, max_queries, device=features.device)
        for i, qlen in enumerate(len_queries):
            query_attn_mask[i, :qlen] = 1
        
        # Run Q-Former
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            attention_mask=query_attn_mask,
            encoder_hidden_states=features,
            encoder_attention_mask=(~padding_mask).long() if padding_mask is not None else None,
            return_dict=True
        )['last_hidden_state']
        
        return query_output

    def calculate_query_lengths(self, features):
        """Calculate dynamic query lengths using speech rate predictor"""
        with torch.no_grad():
            sr_predictions = self.sr_predictor(features[:, ::4, :])
        
        B = features.size(0)
        len_queries = []
        resized_len_list = []
        
        for i in range(B):
            base_queries = features.size(1) / 25 * self.cfg.queries_per_sec
            factor = max(1.0, min(2.0, sr_predictions[i].item()))  # clamp between 1.0-2.0
            adjusted_queries = int(base_queries * factor)
            len_queries.append(max(adjusted_queries, self.cfg.queries_per_sec))
            resized_len_list.append(factor * features.size(1))
        
        return len_queries, resized_len_list

class HubertEncoderWrapper(nn.Module):
    """Wrapper for AV-HuBERT encoder"""
    def __init__(self, w2v_model):
        super().__init__()
        self.w2v_model = w2v_model

    def forward(self, source, padding_mask=None):
        features = self.w2v_model.extract_features(
            source=source,
            padding_mask=padding_mask
        )
        return {
            'encoder_out': features[0],  # T x B x D
            'padding_mask': padding_mask
        }