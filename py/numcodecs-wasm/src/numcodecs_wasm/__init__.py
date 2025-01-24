__all__ = ["WasmCodecMeta"]

import sys
from os import PathLike

from fcbench.codecs import WasmCodecClassLoader


class WasmCodecMeta(type):
    def __new__(cls, clsname, bases, attrs, wasm: PathLike):
        assert len(bases) == 0
        assert sorted(attrs.keys()) == ["__module__", "__qualname__"]

        codec_cls: type = WasmCodecClassLoader.load(
            wasm, sys.modules[attrs["__module__"]]
        )

        assert codec_cls.__name__ == clsname
        assert codec_cls.__qualname__ == attrs["__qualname__"]

        return codec_cls
