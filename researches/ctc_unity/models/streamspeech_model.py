# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import logging
import torch
from typing import OrderedDict, Optional
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
    FairseqEncoderDecoderModel
)
from ..avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoderWrapper

from fairseq_modules.models.speech_to_speech.modules.ctc_decoder import CTCDecoder
from fairseq_modules.models.speech_to_speech.s2s_transformer import S2UTTransformerModel
from ctc_unity.modules.ctc_decoder_with_transformer_layer import (
    CTCDecoderWithTransformerLayer,
)
from fairseq_modules.models.speech_to_speech.modules.stacked_embedding import StackedEmbedding
from fairseq_modules.models.speech_to_speech.modules.transformer_decoder_aug import (
    AugTransformerUnitDecoder,
)
from ctc_unity.modules.transformer_encoder import (
    UniTransformerEncoderNoEmb,
)
from chunk_unity.models.s2s_conformer import ChunkS2UTConformerModel
from fairseq_modules.models.speech_to_speech.s2s_transformer import (
    base_multitask_text_transformer_decoder_arch,
    s2ut_architecture_base,
)
from chunk_unity.models.s2s_transformer import (
    TransformerUnitDecoder,
)
from fairseq_modules.models.transformer import TransformerModelBase
from ctc_unity.modules.transformer_decoder import TransformerDecoder
from ctc_unity.modules.ctc_transformer_unit_decoder import CTCTransformerUnitDecoder

from ..avhubert.hubert_asr import AVHubertAsrConfig, HubertEncoderWrapper
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperModel
from ..modules.sub_model.modules import WhisperEncoderWrapper, Projector, Multimodal_Attention, Speech_Rate_Predictor
from ..modules.sub_model.Qformer import BertConfig, BertLMHeadModel
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)

@dataclass
class MMS_LLaMA_Config(AVHubertAsrConfig):
    llm_path: str = field(
        default='meta-llama/Llama-3.2-3B'
    )
    target_modules: str = field(
        default='q_proj.v_proj.k_proj.o_proj'
    )
    whisper_embed_dim: int = field(
        default=1024, metadata={"help": "whisper embedding dimension"}
    )
    avhubert_embed_dim: int = field(
        default=1024, metadata={"help": "avhubert embedding dimension"}
    )
    llama_embed_dim: int = field(
        default=3072, metadata={"help": "llama embedding dimension"}
    )
    lora_rank: int = field(
        default=16, metadata={"help": "lora_rank"}
    )
    lora_alpha: int = field(
        default=32, metadata={"help": "lora_alpha"}
    )
    modality_fuse: str = field(
        default='concat', metadata={'help': 'fusing two modalities: concat, add, cross-att'}
    )
    ### Speech Q-Former Config ###
    use_qformer: bool = field(
        default=True
    )
    window_level: bool = field(
        default=False
    )
    queries_per_sec: int = field(
        default=4, metadata={"help": "queries_per_sec"}
    )
    qformer_layers: int = field(
        default=2, metadata={"help": "number of qformer layers"}
    )
    qformer_dim: int = field(
        default=1024, metadata={"help": "qformer dim"}
    )
    ### Speech Rate Predictor Config ###
    use_sr_predictor: bool = field(
        default=False
    )
    sr_predictor_layers: int = field(
        default=2, metadata={"help": "number of sr predictor layers"}
    )
    #补充
    #fairseq/fairseq/data/audio/data_cfg.py/S2TDataconfig
    input_feat_per_channel: int = field(
        default=80, metadata={"help": "The dimension of input features (per audio channel)"}
    )
    input_channels: int = field(
        default=1, metadata={"help": "The number of channels in the input audio"}
    )
    target_speaker_embed: Optional[bool] = field(
        default=None, metadata={"help": "Target speaker embedding file (one line per target audio sample)"}
    )
    #SpeechToSpeechTask args
    n_frames_per_step: int = field(
        default=1, metadata={"help": "stacked frames, use 0 for reduced discrete unit sequence"}
    )
    #SpeechToSpeechModel args
    translation_decoder_layers: int = field(
        default=4,
        metadata={
            "help": "num decoder layers in the first-pass translation module",
            "metavar": "N",
        },
    )
    synthesizer: str = field(
        default="transformer",
        metadata={
            "help": "",
            "choices": ["transformer"],
        },
    )
    synthesizer_encoder_layers: int = field(
        default=0,
        metadata={
            "help": "num encoder layers in the second-pass synthesizer module",
            "metavar": "N",
        },
    )
    synthesizer_augmented_cross_attention: bool = field(
        default=False,
        metadata={
            "help": "augmented cross-attention over speech encoder output",
            "action": "store_true",
        },
    )
    load_pretrained_mt_from: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to pretrained s2t transformer model",
        },
    )
    uni_encoder: bool = field(
        default=False,
        metadata={
            "help": "apply unidirectional encoder",
            "action": "store_true",
        },
    )
    ctc_upsample_rate: int = field(
        default=10,
        metadata={
            "metavar": "N",
        },
    )
    depthwise_conv_kernel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "kernel size of depthwise convolution layers",
            "metavar": "N",
        },
    )
    attn_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If not specified uses fairseq MHA. Other valid option is espnet for using conformer",
            "metavar": "STR",
        },
    )
    pos_enc_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Must be specified in addition to attn-type=espnet for rel_pos and rope",
            "metavar": "STR",
        },
    )
    chunk_size: int = field(
        default=-1,
        metadata={
            "help": "chunk size",
            "metavar": "N",
        },
    )
    #MutiTasks args
    #===============base_s2st_transformer_encoder======================
    encoder_freezing_updates: int = field(
        default=0,
        metadata={
            "help": "Number of initial updates where encoder parameters remain frozen",
        },
    )
    input_channels: int = field(
        default=1,
        metadata={
            "help": "Number of input channels for audio features",
        },
    )
    conv_kernel_sizes: str = field(
        default="5,5",
        metadata={
            "help": "Comma-separated convolutional kernel sizes for subsampling layers",
        },
    )
    conv_channels: int = field(
        default=1024,
        metadata={
            "help": "Number of channels in 1D convolutional subsampling layers",
        },
    )
    conv_out_channels: int = field(
        default=256,
        metadata={
            "help": "Output channels for 2D convolutional subsampling layers",
        },
    )
    conv_version: str = field(
        default="s2t_transformer",
        metadata={
            "help": "Version of convolutional architecture (e.g. s2t_transformer, vggnet)",
        },
    )
    encoder_embed_dim: int = field(
        default=512,
        metadata={
            "help": "Encoder embedding dimension",
        },
    )
    encoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={
            "help": "Encoder hidden size for feed-forward networks",
        },
    )
    encoder_layers: int = field(
        default=12,
        metadata={
            "help": "Number of encoder layers",
        },
    )
    encoder_attention_heads: int = field(
        default=8,
        metadata={
            "help": "Number of attention heads in encoder",
        },
    )
    encoder_normalize_before: bool = field(
        default=True,
        metadata={
            "help": "Apply layer normalization before each encoder block",
        },
    )
    no_scale_embedding: bool = field(
        default=False,
        metadata={
            "help": "Disable scaling of embeddings by sqrt(embed_dim)",
        },
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "General dropout probability applied throughout the model",
        },
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout probability for attention weights (defaults to dropout if not set)",
        },
    )
    activation_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Dropout probability after activation functions (defaults to dropout if not set)",
        },
    )
    activation_fn: str = field(
        default="relu",
        metadata={
            "help": "Activation function to use (e.g. relu, gelu, swish)",
        },
    )
    speaker_embed_dim: int = field(
        default=256,
        metadata={
            "help": "Embedding dimension for speaker characteristics",
        },
    )
    #===============base_s2st_transformer_decoder======================
    decoder_embed_dim: int = field(
        default=512,
        metadata={
            "help": "Decoder embedding dimension (defaults to encoder_embed_dim if not set)",
        },
    )
    decoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={
            "help": "Decoder hidden size for feed-forward networks (defaults to encoder_ffn_embed_dim)",
        },
    )
    decoder_layers: int = field(
        default=6,
        metadata={
            "help": "Number of decoder layers",
        },
    )
    decoder_attention_heads: int = field(
        default=8,
        metadata={
            "help": "Number of attention heads in decoder",
        },
    )
    decoder_normalize_before: bool = field(
        default=True,
        metadata={
            "help": "Apply layer normalization before each decoder block",
        },
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={
            "help": "Use learned positional embeddings in decoder",
        },
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma separated list of adaptive softmax cutoff points",
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={
            "help": "Dropout probability for adaptive softmax",
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={
            "help": "Share decoder input and output embeddings",
        },
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Disable token positional embeddings in decoder",
        },
    )
    adaptive_input: bool = field(
        default=False,
        metadata={
            "help": "Use adaptive input embeddings",
        },
    )
    decoder_layerdrop: float = field(
        default=0.0,
        metadata={
            "help": "LayerDrop probability for decoder",
        },
    )
    decoder_output_dim: int = field(
        default=512,
        metadata={
            "help": "Output dimension of decoder (defaults to decoder_embed_dim)",
        },
    )
    decoder_input_dim: int = field(
        default=512,
        metadata={
            "help": "Input dimension of decoder (defaults to decoder_embed_dim)",
        },
    )
    quant_noise_pq: float = field(
        default=0,
        metadata={
            "help": "Quantization noise parameter for product quantization",
        },
    )
    #===============ctc_unity_conformer_architecture_base======================
    conv_version: str = field(
        default="convtransformer",
        metadata={
            "help": "Type of convolutional architecture (e.g. convtransformer, s2t_transformer)",
        },
    )
    attn_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attention mechanism type (e.g. local, sparse, multihead)",
        },
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={
            "help": "Type of positional encoding (absolute, relative, rotary, etc.)",
        },
    )
    # max_source_positions: int = field(
    #     default=6000,
    #     metadata={
    #         "help": "Maximum input sequence length (in frames or tokens)",
    #     },
    # )
    encoder_embed_dim: int = field(
        default=256,
        metadata={
            "help": "Dimension of encoder embeddings",
        },
    )
    encoder_ffn_embed_dim: int = field(
        default=2048,
        metadata={
            "help": "Hidden size of encoder feed-forward networks",
        },
    )
    encoder_attention_heads: int = field(
        default=4,
        metadata={
            "help": "Number of attention heads in encoder layers",
        },
    )
    dropout: float = field(
        default=0.1,
        metadata={
            "help": "Global dropout rate applied throughout the model",
        },
    )
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "Kernel size for depthwise convolutional layers",
        },
    )







@register_model("streamspeech", dataclass=MMS_LLaMA_Config) 
class StreamSpeechModel(FairseqEncoderDecoderModel):
    """
    Direct speech-to-speech translation model with Conformer encoder + MT Transformer decoder + Transformer discrete unit decoder
    """
    def __init__(self, avhubert, whisper, tokenizer, cfg):
        super().__init__() 
        
        self.cfg = cfg
        self.avhubert = avhubert
        self.whisper = whisper

    @classmethod
    def build_multitask_decoder(
        cls,
        args,
        tgt_dict,
        in_dim,
        is_first_pass_decoder,
        decoder_layers,
        decoder_embed_dim,
        decoder_attention_heads,
    ):
        decoder_args = args.decoder_args
        decoder_args.encoder_embed_dim = in_dim
        if args.decoder_type == "transformer":
            if is_first_pass_decoder:
                multitask_text_transformer_decoder_arch(
                    decoder_args,
                    decoder_layers,
                    decoder_embed_dim,
                    decoder_attention_heads,
                )  # 4L
            else:
                base_multitask_text_transformer_decoder_arch(decoder_args)  # 2L
            task_decoder = TransformerDecoder(
                decoder_args,
                tgt_dict,
                embed_tokens=TransformerModelBase.build_embedding(
                    decoder_args,
                    tgt_dict,
                    decoder_args.decoder_embed_dim,
                ),
            )
        elif args.decoder_type == "ctc":
            if getattr(decoder_args, "encoder_layers", 0) == 0:
                task_decoder = CTCDecoder(
                    dictionary=tgt_dict,
                    in_dim=in_dim,
                )
            else:
                task_decoder = CTCDecoderWithTransformerLayer(
                    decoder_args,
                    dictionary=tgt_dict,
                    in_dim=in_dim,
                )
        else:
            raise NotImplementedError(
                "currently only support multitask decoder_type 'transformer', 'ctc'"
            )

        return task_decoder
    @classmethod
    def build_encoder(cls, args):
        # Audio Encoder
        whisper_ = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en").model.encoder
        whisper = WhisperEncoderWrapper(whisper_)

        # Visual Encoder
        arg_overrides = {
            "dropout": args.dropout,
            "activation_dropout": args.activation_dropout,
            "dropout_input": args.dropout_input,
            "attention_dropout": args.attention_dropout,
            "mask_length": args.mask_length,
            "mask_prob": args.mask_prob,
            "mask_selection": args.mask_selection,
            "mask_other": args.mask_other,
            "no_mask_overlap": args.no_mask_overlap,
            "mask_channel_length": args.mask_channel_length,
            "mask_channel_prob": args.mask_channel_prob,
            "mask_channel_selection": args.mask_channel_selection,
            "mask_channel_other": args.mask_channel_other,
            "no_mask_channel_overlap": args.no_mask_channel_overlap,
            "encoder_layerdrop": args.layerdrop,
            "feature_grad_mult": args.feature_grad_mult,
        }

        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # w2v_path = f'{root_dir}/../../pretrained_models/avhubert/large_vox_iter5.pt'
        w2v_path = '/workspace/StreamSpeech/pretrained_models/base_lrs3_iter5.pt'
        print("args.w2v_args : ", args.w2v_args)
        if args.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(
                w2v_path, arg_overrides
            )
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            args.w2v_args = w2v_args
        else:
            state = None
            w2v_args = args.w2v_args
            if isinstance(w2v_args, Namespace):
                args.w2v_args = w2v_args = convert_namespace_to_omegaconf(
                    w2v_args
                )
        print("args.w2v_args : ", args.w2v_args)
        assert args.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        ########### 修改input_modality项 ##########
        # OmegaConf.set_struct(w2v_args.task, False)   # 关闭只读
        # w2v_args.task.pop("input_modality", None)
        # w2v_args.task.modalities = ["audio", "video"]
        # OmegaConf.set_struct(w2v_args.task, True)

        w2v_args.task.data = args.data
        print(w2v_args.task)
       
        task_pretrain = tasks.setup_task(w2v_args.task)
        print(task_pretrain)
        exit()
        if state is not None:
            task_pretrain.load_state_dict(state['task_state'])

        encoder_ = task_pretrain.build_model(w2v_args.model)
        avhubert = HubertEncoderWrapper(encoder_)
        pass

    @classmethod
    def build_decoder(cls, args, tgt_dict, aug_attn=False):
        num_embeddings = len(tgt_dict)
        padding_idx = tgt_dict.pad()
        embed_tokens = StackedEmbedding(
            num_embeddings,
            args.decoder_embed_dim,
            padding_idx,
            num_stacked=args.n_frames_per_step,
        )

        _args = copy.deepcopy(args)
        _args.encoder_embed_dim = args.decoder_embed_dim

        decoder_cls = CTCTransformerUnitDecoder  # AugTransformerUnitDecoder if aug_attn else TransformerUnitDecoder
        return decoder_cls(
            _args,
            tgt_dict,
            embed_tokens,
        )

    @classmethod
    def build_model(cls, args, task):
        #ChunkS2SConformerEncoder
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(
            args,
            task.tgt_dict,
            aug_attn=getattr(args, "synthesizer_augmented_cross_attention", False),
        )
        base_model = cls(encoder, decoder)

        base_model.t2u_augmented_cross_attn = getattr(
            args, "synthesizer_augmented_cross_attention", False
        )

        # set up multitask decoders
        base_model.mt_task_name = None
        base_model.multitask_decoders = {}
        has_first_pass_decoder = False
        for task_name, task_obj in task.multitask_tasks.items():
            if task_obj.is_first_pass_decoder:
                has_first_pass_decoder = True
                base_model.mt_task_name = task_name

            in_dim = (
                args.encoder_embed_dim
                if task_obj.args.input_from == "encoder"
                else args.decoder_embed_dim
            )
            task_decoder = cls.build_multitask_decoder(
                task_obj.args,
                task_obj.target_dictionary,
                in_dim,
                task_obj.is_first_pass_decoder,
                getattr(args, "translation_decoder_layers", 4),
                getattr(args, "decoder_embed_dim", 256),
                getattr(args, "decoder_attention_heads", 4),
            )

            setattr(base_model, f"{task_name}_decoder", task_decoder)
            decoder_model_cls = (
                FairseqEncoderModel
                if task_obj.args.decoder_type == "ctc"
                else FairseqLanguageModel
            )
            base_model.multitask_decoders[task_name] = decoder_model_cls(
                getattr(base_model, f"{task_name}_decoder")
            )

        assert has_first_pass_decoder, "set at least one intermediate non-CTC decoder"

        # set up encoder on top of the auxiliary MT decoder
        if getattr(args, "synthesizer_encoder_layers", 0) > 0:
            base_model.synthesizer_encoder = cls.build_text_encoder(args)
        else:
            base_model.synthesizer_encoder = None

        if getattr(args, "load_pretrained_mt_from", None):
            state_dict = checkpoint_utils.load_checkpoint_to_cpu(
                args.load_pretrained_mt_from
            )["model"]
            encoder_state_dict = OrderedDict()
            decoder_state_dict = OrderedDict()
            for key in state_dict.keys():
                if key.startswith("encoder"):
                    subkey = key[len("encoder") + 1 :]
                    encoder_state_dict[subkey] = state_dict[key]
                elif key.startswith("decoder"):
                    decoder_state_dict[key] = state_dict[key]
            base_model.encoder.load_state_dict(encoder_state_dict)
            base_model.multitask_decoders[base_model.mt_task_name].load_state_dict(
                decoder_state_dict
            )
            logger.info(
                f"Successfully load pretrained Conformer from {args.load_pretrained_mt_from}."
            )

        return base_model

    @classmethod
    def build_text_encoder(cls, args):
        _args = copy.deepcopy(args)
        _args.encoder_layers = args.synthesizer_encoder_layers
        _args.encoder_embed_dim = args.decoder_embed_dim
        _args.encoder_ffn_embed_dim = args.decoder_ffn_embed_dim
        _args.encoder_attention_heads = args.decoder_attention_heads
        _args.encoder_normalize_before = True
        return UniTransformerEncoderNoEmb(_args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        prev_output_tokens_mt,
        streaming_config=None,
        tgt_speaker=None,
        return_all_hiddens=False,
    ):
        mt_decoder = getattr(self, f"{self.mt_task_name}_decoder")
        #encoder_out.dim = torch.Size([264, 16, 256])
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            tgt_speaker=tgt_speaker,
            return_all_hiddens=return_all_hiddens,
        )
        if streaming_config is not None:
            asr_decoder = getattr(self, "source_unigram_decoder")
            asr_ctc_out = asr_decoder(encoder_out["encoder_out"][0].detach())
            asr_probs = self.get_normalized_probs(
                [asr_ctc_out["encoder_out"].transpose(0, 1)], log_probs=False
            )
            asr_repeat = (
                torch.cat(
                    (
                        torch.zeros(
                            (asr_probs.size(0), 1, asr_probs.size(-1) - 1),
                            device=asr_probs.device,
                        ),
                        asr_probs[:, :-1, 1:],
                    ),
                    dim=1,
                )
                * asr_probs[:, :, 1:]
            )
            asr_repeat = asr_repeat.sum(dim=-1, keepdim=False)
            asr_blank = asr_probs[:, :, 0]
            asr_not_blank = 1 - (asr_repeat + asr_blank).detach()

            st_decoder = getattr(self, "ctc_target_unigram_decoder")
            st_ctc_out = st_decoder(encoder_out["encoder_out"][0].detach())
            st_probs = self.get_normalized_probs(
                [st_ctc_out["encoder_out"].transpose(0, 1)], log_probs=False
            )
            st_repeat = (
                torch.cat(
                    (
                        torch.zeros(
                            (st_probs.size(0), 1, st_probs.size(-1) - 1),
                            device=st_probs.device,
                        ),
                        st_probs[:, :-1, 1:],
                    ),
                    dim=1,
                )
                * st_probs[:, :, 1:]
            )
            st_repeat = st_repeat.sum(dim=-1, keepdim=False)
            st_blank = st_probs[:, :, 0]
            st_not_blank = 1 - (st_repeat + st_blank).detach()

            streaming_mask = self.build_streaming_mask(
                asr_not_blank,
                st_not_blank,
                prev_output_tokens_mt,
                streaming_config["k1"],
                streaming_config["n1"],
                streaming_config["n1"],
            )
            streaming_config["streaming_mask"] = streaming_mask

        # 1. MT decoder
        mt_decoder_out = mt_decoder(
            prev_output_tokens_mt,
            encoder_out=encoder_out,
            streaming_config=streaming_config,
        )
        x = mt_decoder_out[1]["inner_states"][-1]
        if mt_decoder.layer_norm is not None:
            x = mt_decoder.layer_norm(x)

        mt_decoder_padding_mask = None
        if prev_output_tokens_mt.eq(mt_decoder.padding_idx).any():
            mt_decoder_padding_mask = prev_output_tokens_mt.eq(mt_decoder.padding_idx)

        # 2. T2U encoder
        if self.synthesizer_encoder is not None:
            t2u_encoder_out = self.synthesizer_encoder(
                x,
                mt_decoder_padding_mask,
                return_all_hiddens=return_all_hiddens,
            )
        else:
            t2u_encoder_out = {
                "encoder_out": [x],  # T x B x C
                "encoder_padding_mask": [mt_decoder_padding_mask],  # B x T
            }

        # 3. T2U decoder
        if self.t2u_augmented_cross_attn:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                encoder_out_aug=t2u_encoder_out,
            )
        else:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=t2u_encoder_out,
                streaming_config=(
                    {
                        "src_wait": int(streaming_config["k2"]),
                        "src_step": int(streaming_config["n2"]),
                    }
                    if streaming_config is not None
                    else None
                ),
            )
            
        if return_all_hiddens:
            decoder_out[-1]["encoder_states"] = encoder_out["encoder_states"]
            decoder_out[-1]["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ]
        decoder_out[-1]["mt_decoder_out"] = mt_decoder_out
        return decoder_out

    def build_streaming_mask(self, asr, st, y, src_wait, src_step, tgt_step):
        tgt_len = y.size(1)
        bsz, src_len = st.size()
        idx = torch.arange(0, tgt_len, device=st.device).unsqueeze(0).unsqueeze(2)
        idx = (idx // tgt_step + 1) * src_step + src_wait
        idx = idx.clamp(1, src_len)
        tmp = st.cumsum(dim=-1).unsqueeze(1)
        mask = tmp >= idx
        tmp2 = mask.int() * asr.round().unsqueeze(1)
        tmp2[:, :, -1] = 1
        idx2 = tmp2.max(dim=-1, keepdim=True)[1].clamp(1, src_len)
        if self.encoder.chunk:
            chunk_size = self.encoder.chunk_size
            idx2 = (idx2 // chunk_size + 1) * chunk_size
            idx2 = idx2.clamp(1, src_len)
        tmp3 = torch.arange(0, src_len, device=st.device).unsqueeze(0).unsqueeze(1)

        return tmp3 >= idx2
    
    # def upgrade_state_dict_named(self, state_dict, name):
    #     super().upgrade_state_dict_named(state_dict, name)
    #     return state_dict

    # def set_num_updates(self, num_updates):
    #     """Set the number of parameters updates."""
    #     super().set_num_updates(num_updates)
    #     self.num_updates = num_updates
        
    # def state_dict(self):
    #     old_state = super().state_dict()
    #     state = {k:v for k,v in old_state.items() if k not in self.freeze_params}
    #     return state
    
    # def load_state_dict(self,state,**kwargs):
    #     super().load_state_dict(state, strict=False)   


# @register_model_architecture(model_name="streamspeech", arch_name="streamspeech")
# def ctc_unity_conformer_architecture_base(args):
    # args.conv_version = getattr(args, "conv_version", "convtransformer")
    # args.attn_type = getattr(args, "attn_type", None)
    # args.pos_enc_type = getattr(args, "pos_enc_type", "abs")
    # args.max_source_positions = getattr(args, "max_source_positions", 6000)
    # args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    # args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    # args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    # args.dropout = getattr(args, "dropout", 0.1)
    # args.encoder_layers = getattr(args, "encoder_layers", 16)
    # args.depthwise_conv_kernel_size = getattr(args, "depthwise_conv_kernel_size", 31)
#     s2ut_architecture_base(args)

