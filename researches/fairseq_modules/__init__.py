from .data import *
# from .data.audio.waveform_transforms import *
from fairseq_modules.models.speech_to_speech import *
from fairseq_modules.models.text_to_speech import *
from fairseq_modules.models.speech_to_text import *
# from fairseq_modules.models.transformer import *
# from .location_attention import *
# from .lstm_cell_with_zoneout import *

from hydra.core.config_store import ConfigStore
from fairseq_modules.models.fairseq_decoder import FairseqDecoder
from fairseq_modules.models.fairseq_encoder import FairseqEncoder
from fairseq_modules.models.fairseq_model import (
    BaseFairseqModel,
    FairseqEncoderDecoderModel,
    FairseqEncoderModel,
    FairseqLanguageModel,
    FairseqModel,
    FairseqMultiModel,
)

MODEL_REGISTRY = {}
MODEL_DATACLASS_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


__all__ = [
    "BaseFairseqModel",
    "FairseqDecoder",
    "FairseqEncoder",
    "FairseqEncoderDecoderModel",
    "FairseqEncoderModel",
    "FairseqLanguageModel",
    "FairseqModel",
    "FairseqMultiModel",
]

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

print("fairseq plugins loaded...")

import os
import importlib

def register_model(name, dataclass=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            return MODEL_REGISTRY[name]

        if not issubclass(cls, BaseFairseqModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseFairseqModel".format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls
        if dataclass is not None and not issubclass(dataclass, FairseqDataclass):
            raise ValueError(
                "Dataclass {} must extend FairseqDataclass".format(dataclass)
            )

        cls.__dataclass = dataclass
        if dataclass is not None:
            MODEL_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="model", node=node, provider="fairseq")

            @register_model_architecture(name, name)
            def noop(_):
                pass

        return cls

    return register_model_cls


def register_model_architecture(model_name, arch_name):
    """
    New model architectures can be added to fairseq with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture('lstm', 'lstm_luong_wmt_en_de')
        def lstm_luong_wmt_en_de(cfg):
            args.encoder_embed_dim = getattr(cfg.model, 'encoder_embed_dim', 1000)
            (...)

    The decorated function should take a single argument *cfg*, which is a
    :class:`omegaconf.DictConfig`. The decorated function should modify these
    arguments in-place to match the desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn):
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                "Cannot register model architecture for unknown model type ({})".format(
                    model_name
                )
            )
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError(
                "Cannot register duplicate model architecture ({})".format(arch_name)
            )
        if not callable(fn):
            raise ValueError(
                "Model architecture must be callable ({})".format(arch_name)
            )
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_NAME_REGISTRY[arch_name] = model_name
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq_modules." + file_name)
