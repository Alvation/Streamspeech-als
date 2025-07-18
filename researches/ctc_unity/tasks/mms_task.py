# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os, glob
import sys
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from fairseq_modules.tasks.speech_to_speech import SpeechToSpeechTask
from ctc_unity.datasets.speech_to_speech_dataset_modified import (
    SpeechToSpeechDatasetModifiedCreator,
)
from ctc_unity.datasets.speech_to_speech_data_cfg_modified import S2SDataConfigModified
from dataclasses import dataclass, field
from fairseq import metrics, search
from fairseq.data import Dictionary, encoders
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING, II
import numpy as np
from argparse import Namespace

DBG=True if len(sys.argv) == 1 else False

if DBG:
    from ctc_unity.datasets.mms_dataset import mms_llama_dataset
else:
    from ctc_unity.datasets.mms_dataset import mms_llama_dataset

logger = logging.getLogger(__name__)

@dataclass
class MMS_LLaMA_TrainingConfig(FairseqDataclass):
    # public
    data: str = field(
        default=MISSING, metadata={"help": "path to data directory"}
    )
    #speech_to_speech args begin
    config_yaml: str = field(
        default="config.yaml",
        metadata={
            "help": (
                "Configuration YAML filename (under manifest root)"
            )
        },
    )
    multitask_config_yaml: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Configuration YAML filename for the multitasks (under manifest root)"
            )
        },
    )
    max_source_positions: int = field(
        default=6000,
        metadata={
            "help": (
                "max number of tokens in the source sequence"
            )
        },
    )
    max_target_positions: int = field(
        default=1024,
        metadata={
            "help": (
                "max number of tokens in the target sequence"
            )
        },
    )
    target_is_code: bool = field(
        default=False,
        metadata={
            "help": (
                "set if target is discrete unit instead of spectrogram"
            )
        },
    )
    target_code_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "# discrete units"
            )
        },
    )
    n_frames_per_step: int = field(
        default=1,
        metadata={
            "help": (
                "# stacked frames, use 0 for reduced discrete unit sequence"
            )
        },
    )
    eval_inference: bool = field(
        default=True,
        metadata={
            "help": (
                "Configuration YAML filename (under manifest root)"
            )
        },
    )
    eval_args: Dict[str, Any] = field(
        default_factory=lambda: {},  # 默认空字典
        metadata={"help": "..."},
    )
    eos_prob_threshold: float = field(
        default=0.5,
    )
    mcd_normalize_type: str = field(
        default="targ",
        metadata={
            "help": "Type of MCD normalization (targ, pred, path)",
            "choices": ["targ", "pred", "path"],  # 相当于 argparse 的 choices
        },
    )
    vocoder: str = field(
        default="griffin_lim",
        metadata={
            "help": "Type of MCD normalization (targ, pred, path)",
            "choices": ["griffin_lim", "hifigan", "code_hifigan"],  # 相当于 argparse 的 choices
        },
    )
    spec_bwd_max_iter: int = field(
        default=8,
        metadata={
            "help": "Maximum number of iterations for spectral backward pass",
        },
    )
    infer_target_lang: str = field(
        default="",
        metadata={
            "help": "Target language for inference",
        },
    )
    #speech_to_speech args end
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    llm_path: str = field(
        default=MISSING, metadata={"help": "path to llama checkpoint"}
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to keep in training"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to keep in training"},
    )
    max_trim_sample_size: Optional[int] = field(
        default=II("task.max_sample_size"),
        metadata={"help": "max sample size to trim to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    pdb: Optional[bool] = field(
        default=False,
        metadata={"help": "pdb"},
    )
    stack_order_audio: int = field(
        default=1,
        metadata={"help": "concatenate n consecutive audio frames for one step"},
    )
    skip_verify: Optional[bool] = field(
        default=False,
        metadata={"help": "skip verifying label-audio alignment"},
    )
    image_aug: bool = field(default=False, metadata={'help': 'image data augmentation'})
    image_crop_size: int = field(
        default=88, metadata={"help": "image ROI size"})
    image_mean: float = field(
        default=0.421, metadata={"help": "image mean"})
    image_std: float = field(
        default=0.165, metadata={"help": "image std"})
    modalities: Optional[List[str]] = field(default_factory=lambda: ["audio", "video"], metadata={'help': 'modalities to load'})
    is_s2s: bool=field(default=False, metadata={'help': 'seq2seq fine-tuning only'})
    tokenizer_bpe_name: Optional[str] = field(default=None, metadata={'help': 'tokenizer model name'})
    tokenizer_bpe_model: Optional[str] = field(default=None, metadata={'help': 'tokenizer model path'})
    noise_wav: Optional[str] = field(default=None, metadata={'help': 'manifest of noise wav files (one wav file path per line)'})
    noise_prob: float = field(default=0, metadata={'help': 'noise probability'})
    noise_snr: Optional[str] = field(default='0', metadata={'help': 'noise SNR in audio'})
    snr_target: Optional[str] = field(default=None, metadata={'help': 'noise SNR in audio'})
    noise_num: int = field(default=1, metadata={'help': 'number of noise wav files to mix'})
    fine_tuning: bool = field(default=False, metadata={"help": "set to true if fine-tuning AV-Hubert"})

@register_task("MMS_LLaMA_training", dataclass=MMS_LLaMA_TrainingConfig)
class MMS_LLaMA_TrainingTask(SpeechToSpeechTask):
# class MMS_LLaMA_TrainingTask(FairseqTask):
    
    def __init__(
        self, args, tgt_dict, infer_tgt_lang_id=None):

        tgt_blank_index = tgt_dict.add_symbol("<s>")
        self.tgt_dict = tgt_dict
        self.tgt_dict.blank_index = tgt_blank_index
        super().__init__(args, tgt_dict, infer_tgt_lang_id)
        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"MMS_LLaMA_TrainingTask Config {args}")

        self.fine_tuning = args.fine_tuning    
        # self.blank_symbol = "<s>"
    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None
    
    @property
    def dictionaries(self) -> List[Dictionary]:
        return None

    # @classmethod
    # def setup_task(
    #     cls, cfg: MMS_LLaMA_TrainingConfig, **kwargs
    # ) -> "MMS_LLaMA_TrainingTask":
    #     if cfg.pdb:
    #         import pdb
    #         pdb.set_trace()
    #     return cls(cfg)

    def get_label_dir(self) -> str:
        if self.args.label_dir is None:
            return self.args.data
        return self.args.label_dir

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.args.data}/{split}.tsv"
        logger.info(f"Using tokenizer")
        paths = [
            f"{self.get_label_dir()}/{split}.{l}" for l in self.args.labels
        ]
        image_aug = self.args.image_aug if split == 'train' else False
        noise_num = self.args.noise_num # 
        self.datasets[split] = mms_llama_dataset(
            manifest,
            sample_rate=self.args.sample_rate,
            llm_path=self.args.llm_path,
            label_paths=paths,
            label_rates=self.args.label_rate,
            max_keep_sample_size=self.args.max_sample_size,
            min_keep_sample_size=self.args.min_sample_size,
            max_sample_size=self.args.max_trim_sample_size,
            pad_audio=self.args.pad_audio,
            normalize=self.args.normalize,
            store_labels=True,
            random_crop=self.args.random_crop,
            single_target=self.args.single_target,
            stack_order_audio=self.args.stack_order_audio,
            skip_verify=self.args.skip_verify,
            image_mean=self.args.image_mean,
            image_std=self.args.image_std,
            image_crop_size=self.args.image_crop_size,
            image_aug=image_aug,
            modalities=self.args.modalities,
            is_s2s=self.args.is_s2s,
            noise_fn=self.args.noise_wav,
            noise_prob=self.args.noise_prob,
            snr_target=self.args.snr_target
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices
    
    def build_generator_dual_decoder(
        self,
        models,
        args,
        extra_gen_cls_kwargs=None,
    ):
        from ctc_unity.sequence_generator_multi_decoder_ctc import (
            CTCMultiDecoderSequenceGenerator,
        )

        return CTCMultiDecoderSequenceGenerator(
            models,
            self.target_dictionary,
            self.target_dictionary_mt,
            beam_size=max(1, getattr(args, "beam", 1)),
            beam_size_mt=max(1, getattr(args, "beam_mt", 1)),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            max_len_a_mt=getattr(args, "max_len_a_mt", 0),
            max_len_b_mt=getattr(args, "max_len_b_mt", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            **extra_gen_cls_kwargs,
        )
