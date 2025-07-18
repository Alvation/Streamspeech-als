import os
import importlib

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        # 源代码
        # importlib.import_module("translatotron.datasets." + file_name)
        importlib.import_module("ctc_unity.datasets." + file_name)
