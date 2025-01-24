import importlib.resources

from numcodecs_wasm import WasmCodecMeta


class %CodecName%(
    metaclass=WasmCodecMeta,
    wasm=importlib.resources.files("fcbench")
    .joinpath("data")
    .joinpath("codecs")
    .joinpath("%wasm-file%.wasm"),
):
    pass
