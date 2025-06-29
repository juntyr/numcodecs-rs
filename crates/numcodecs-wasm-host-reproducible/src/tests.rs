use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use numcodecs::{Codec, DynCodecType};

use crate::ReproducibleWasmCodecType;

// codecs don't need to preallocate the full 4GB wasm32 memory space, but
//  still give them a reasonable static allocation for better codegen
const WASM_PAGE_SIZE: u32 = 0x10000 /* 64kiB */;
const MEMORY_RESERVATION: u32 = WASM_PAGE_SIZE * 16 * 64 /* 64MiB */;
const MEMORY_GUARD_SIZE: u32 = WASM_PAGE_SIZE * 16 * 64 /* 64MiB */;
const MEMORY_RESERVATION_FOR_GROWTH: u32 = WASM_PAGE_SIZE * 16 * 64 /* 64MiB */;

#[test]
fn codec_roundtrip() {
    // keep in sync with numcodecs-wasm
    let mut config = wasmtime::Config::new();
    config
        .cranelift_nan_canonicalization(true)
        .cranelift_opt_level(wasmtime::OptLevel::Speed)
        .memory_reservation(u64::from(MEMORY_RESERVATION))
        .memory_guard_size(u64::from(MEMORY_GUARD_SIZE))
        .memory_reservation_for_growth(u64::from(MEMORY_RESERVATION_FOR_GROWTH))
        // WASM feature restrictions, follows the feature validation in
        //  numcodecs_wasm_host_reproducible::engine::ValidatedModule::new
        .wasm_bulk_memory(true)
        .wasm_custom_page_sizes(false)
        .wasm_extended_const(false)
        .wasm_function_references(false)
        .wasm_gc(false)
        .wasm_memory64(false)
        .wasm_multi_memory(true)
        .wasm_multi_value(true)
        .wasm_reference_types(false)
        .wasm_relaxed_simd(false)
        .wasm_simd(true)
        .wasm_tail_call(false)
        .wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Enable)
        // wasmtime is compiled without the `threads` feature
        // .wasm_threads(false)
        .wasm_wide_arithmetic(true);

    wasmtime::Cache::from_file(None)
        .map(|cache| config.cache(Some(cache)))
        .unwrap();

    let engine = wasmtime_runtime_layer::Engine::new(wasmtime::Engine::new(&config).unwrap());

    let ty = match ReproducibleWasmCodecType::new(engine, include_bytes!("../tests/round.wasm")) {
        Ok(ty) => ty,
        Err(err) => panic!(
            "ReproducibleWasmCodecType::new:\n===\n{err}\n===\n{err:?}\n===\n{err:#}\n===\n{err:#?}\n===\n"
        ),
    };

    assert_eq!(ty.codec_id(), "round.rs");

    let codec = match ty.codec_from_config(serde_json::json!({ "precision": 1 })) {
        Ok(codec) => codec,
        Err(err) => panic!(
            "ReproducibleWasmCodecType::codec_from_config:\n===\n{err}\n===\n{err:?}\n===\n{err:#}\n===\n{err:#?}\n===\n"
        ),
    };

    let data = Array::random((256, 256), Normal::new(0.0, 1.0).unwrap());

    let encoded = match codec.encode(numcodecs::AnyArray::F64(data.clone().into_dyn()).into_cow()) {
        Ok(encoded) => encoded,
        Err(err) => panic!(
            "ReproducibleWasmCodec::encode:\n===\n{err}\n===\n{err:?}\n===\n{err:#}\n===\n{err:#?}\n===\n"
        ),
    };

    let mut decode_into = numcodecs::AnyArray::F64(Array::zeros((256, 256)).into_dyn());

    match codec.decode_into(encoded.view(), decode_into.view_mut()) {
        Ok(()) => (),
        Err(err) => panic!(
            "ReproducibleWasmCodec::decode_into:\n===\n{err}\n===\n{err:?}\n===\n{err:#}\n===\n{err:#?}\n===\n"
        ),
    };

    let decoded = match codec.decode(encoded.into_cow()) {
        Ok(decoded) => decoded,
        Err(err) => panic!(
            "ReproducibleWasmCodec::decode:\n===\n{err}\n===\n{err:?}\n===\n{err:#}\n===\n{err:#?}\n===\n"
        ),
    };

    assert_eq!(decoded, decode_into);
}
