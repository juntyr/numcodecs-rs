[![CI Status]][workflow] [![PyPi Release]][pypi]

[CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
[workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain

[PyPi Release]: https://img.shields.io/pypi/v/numcodecs-wasm.svg
[pypi]: https://pypi.python.org/pypi/numcodecs-wasm

# numcodecs-wasm

[`numcodecs`][numcodecs] compression for codecs compiled to WebAssembly.

[`numcodecs_wasm`][numcodecs_wasm]: provides the [`WasmCodecMeta`][numcodecs_wasm.WasmCodecMeta] meta class to load a codec from a WebAssembly component into a fresh Python class.

The following Python packages, all published independently on PyPi, use [`numcodecs_wasm`][numcodecs_wasm] to provide their codecs:

- [`numcodecs_wasm_asinh`][numcodecs_wasm_asinh]: $\text{asinh}(x)$ codec
- [`numcodecs_wasm_bit_round`][numcodecs_wasm_bit_round]: bit rounding codec
- [`numcodecs_wasm_ebcc`][numcodecs_wasm_ebcc]: EBCC codec
- [`numcodecs_wasm_fixed_offset_scale`][numcodecs_wasm_fixed_offset_scale]: $\frac{x - o}{s}$ codec
- [`numcodecs_wasm_fourier_network`][numcodecs_wasm_fourier_network]: fourier feature neural network codec
- [`numcodecs_wasm_identity`][numcodecs_wasm_identity]: identity codec
- [`numcodecs_wasm_jpeg2000`][numcodecs_wasm_jpeg2000]: JPEG 2000 codec
- [`numcodecs_wasm_linear_quantize`][numcodecs_wasm_linear_quantize]: linear quantization codec
- [`numcodecs_wasm_log`][numcodecs_wasm_log]: $\ln(x)$ codec
- [`numcodecs_wasm_pco`][numcodecs_wasm_pco]: pcodec
- [`numcodecs_wasm_qpet_sperr`][numcodecs_wasm_qpet_sperr]: QPET-SPERR codec
- [`numcodecs_wasm_random_projection`][numcodecs_wasm_random_projection]: random projection codec
- [`numcodecs_wasm_reinterpret`][numcodecs_wasm_reinterpret]: binary reinterpret codec
- [`numcodecs_wasm_round`][numcodecs_wasm_round]: rounding codec
- [`numcodecs_wasm_sperr`][numcodecs_wasm_sperr]: SPERR codec
- [`numcodecs_wasm_stochastic_rounding`][numcodecs_wasm_stochastic_rounding]: stochastic rounding codec
- [`numcodecs_wasm_swizzle_reshape`][numcodecs_wasm_swizzle_reshape]: array axis swizzle and reshape codec
- [`numcodecs_wasm_sz3`][numcodecs_wasm_sz3]: SZ3 codec
- [`numcodecs_wasm_tthresh`][numcodecs_wasm_tthresh]: Tthresh codec
- [`numcodecs_wasm_uniform_noise`][numcodecs_wasm_uniform_noise]: uniform noise codec
- [`numcodecs_wasm_zfp`][numcodecs_wasm_zfp]: ZFP codec
- [`numcodecs_wasm_zlib`][numcodecs_wasm_zlib]: zlib codec
- [`numcodecs_wasm_zstd`][numcodecs_wasm_zstd]: Zstdandard codec

## Funding

The `numcodecs-wasm` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
