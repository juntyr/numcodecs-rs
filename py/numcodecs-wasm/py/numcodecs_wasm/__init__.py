__all__ = ["WasmCodecMeta", "create_codec_class"]

import sys
from types import ModuleType

import numcodecs.abc

from ._wasm import _create_codec_class, _read_codec_instruction_counter


class WasmCodecMeta(type):
    def __new__(cls, clsname, bases, attrs, wasm: bytes):
        assert len(bases) == 0
        assert sorted(attrs.keys()) == ["__module__", "__qualname__"]

        codec_cls = create_codec_class(sys.modules[attrs["__module__"]], wasm)

        assert codec_cls.__name__ == clsname
        assert codec_cls.__qualname__ == attrs["__qualname__"]

        return codec_cls


def create_codec_class(module: ModuleType, wasm: bytes) -> type[numcodecs.abc.Codec]:
    return _create_codec_class(module, wasm)


def read_instruction_counter(codec: numcodecs.abc.Codec) -> int:
    return _read_codec_instruction_counter(codec)
