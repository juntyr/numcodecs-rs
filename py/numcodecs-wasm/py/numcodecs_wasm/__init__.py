"""
[`numcodecs`][numcodecs] compression for codecs compiled to WebAssembly.

`numcodecs-wasm` provides the [`WasmCodecMeta`][numcodecs_wasm.WasmCodecMeta]
meta class to load a codec from a WebAssembly component into a fresh Python
class.
"""

__all__ = [
    "WasmCodecMeta",
    "create_codec_class",
    "WasmCodecInstructionCounterObserver",
    "read_codec_instruction_counter",
]

import sys
from collections import defaultdict
from collections.abc import Mapping
from typing import Callable, Optional
from typing_extensions import Buffer  # MSPV 3.12
from types import ModuleType, MappingProxyType

import numpy as np
from numcodecs.abc import Codec
from numcodecs_observers.abc import CodecObserver
from numcodecs_observers.hash import HashableCodec

from ._wasm import _create_codec_class, _read_codec_instruction_counter


class WasmCodecMeta(type):
    """
    Meta class to create a [`Codec`][numcodecs.abc.Codec] class from the WebAssembly component `wasm`.

    Parameters
    ----------
    wasm : bytes
        Bytes of the WebAssembly component, from which the class is created.
    """

    def __init__(self, clsname, bases, attrs, wasm: bytes):
        pass

    def __new__(cls, clsname, bases, attrs, wasm: bytes):
        assert len(bases) == 0

        codec_cls = create_codec_class(sys.modules[attrs["__module__"]], wasm)

        assert codec_cls.__name__ == clsname
        assert codec_cls.__qualname__ == attrs["__qualname__"]

        return codec_cls


def create_codec_class(module: ModuleType, wasm: bytes) -> type[Codec]:
    """
    Create a fresh [`Codec`][numcodecs.abc.Codec] class from the WebAssembly component `wasm`.
    The class will be created into the provided `module`.

    Parameters
    ----------
    module : ModuleType
        Module into which the fresh class will be created.
    wasm : bytes
        Bytes of the WebAssembly component, from which the class is created.

    Returns
    -------
    cls : type[Codec]
        Fresh [`Codec`][numcodecs.abc.Codec] class.
    """

    return _create_codec_class(module, wasm)


class WasmCodecInstructionCounterObserver(CodecObserver):
    """
    Observer that measures the number of executed instructions it takes to encode / decode.

    The list of measurements are exposed in the
    [`encode_instructions`][numcodecs_wasm.WasmCodecInstructionCounterObserver.encode_instructions]
    and
    [`decode_instructions`][numcodecs_wasm.WasmCodecInstructionCounterObserver.decode_instructions]
    properties.
    """

    _encode_instructions: defaultdict[HashableCodec, list[float]]
    _decode_instructions: defaultdict[HashableCodec, list[float]]

    def __init__(self):
        self._encode_instructions = defaultdict(list)
        self._decode_instructions = defaultdict(list)

    @property
    def encode_instructions(self) -> Mapping[HashableCodec, list[float]]:
        """
        Per-codec-instance measurements of the number of executed instructions
        it takes to encode.
        """

        return MappingProxyType(self._encode_instructions)

    @property
    def decode_instructions(self) -> Mapping[HashableCodec, list[float]]:
        """
        Per-codec-instance measurements of the number of executed instructions
        it takes to decode.
        """

        return MappingProxyType(self._decode_instructions)

    @np.errstate(divide="raise", over="ignore", under="ignore", invalid="raise")
    def observe_encode(self, codec: Codec, buf: Buffer) -> Callable[[Buffer], None]:
        # Check if the codec supports reading the instruction counter,
        #  otherwise return a no-op observer
        try:
            instructions_before = np.uint64(_read_codec_instruction_counter(codec))
        except TypeError:
            return lambda encoded: None

        def post_encode(encoded: Buffer) -> None:
            instructions_after = np.uint64(_read_codec_instruction_counter(codec))

            self._encode_instructions[HashableCodec(codec)].append(
                int(instructions_after - instructions_before)
            )

        return post_encode

    def observe_decode(
        self, codec: Codec, buf: Buffer, out: Optional[Buffer] = None
    ) -> Callable[[Buffer], None]:
        # Check if the codec supports reading the instruction counter,
        #  otherwise return a no-op observer
        try:
            instructions_before = np.uint64(read_codec_instruction_counter(codec))
        except TypeError:
            return lambda decoded: None

        def post_decode(decoded: Buffer) -> None:
            instructions_after = np.uint64(read_codec_instruction_counter(codec))

            self._decode_instructions[HashableCodec(codec)].append(
                int(instructions_after - instructions_before)
            )

        return post_decode


def read_codec_instruction_counter(codec: Codec) -> int:
    """
    Read the instruction counter of the `codec`.

    Parameters
    ----------
    codec : Codec
        The codec whose instruction counter is read.

    Returns
    -------
    instructions : int
        The number of instructions executed by this codec thus far.

    Raises
    ------
    TypeError
        If the `codec` does not provide an instruction counter.
    """

    return _read_codec_instruction_counter(codec)
