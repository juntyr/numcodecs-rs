__all__ = ["%CodecName%"]

import importlib.resources
import sys

from numcodecs_wasm import WasmCodecMeta


with importlib.resources.as_file(
    importlib.resources.files(sys.modules[__name__]).joinpath("codec.wasm")
) as wasm:
    class %CodecName%(metaclass=WasmCodecMeta, wasm=wasm):
        pass
