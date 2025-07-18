# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .s2s_conformer import *  # noqa
from .s2s_conformer_translatotron2 import *  # noqa
from .s2s_conformer_unity import *  # noqa
from .s2s_transformer import *  # noqa
# import os
# import importlib

# # automatically import any Python files in the criterions/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith(".py") and not file.startswith("_"):
#         file_name = file[: file.find(".py")]
#         importlib.import_module("ctc_unity.speech_to_speech." + file_name)