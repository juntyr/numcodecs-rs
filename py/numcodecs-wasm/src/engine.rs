use pyo3::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
pub type Engine = wasmtime_runtime_layer::Engine;

#[cfg(not(target_arch = "wasm32"))]
pub fn default_engine(py: Python) -> Result<Engine, PyErr> {
    use pyo3_error::PyErrChain;

    // codecs don't need to preallocate the full 4GB wasm32 memory space, but
    //  still give them a reasonable static allocation for better codegen
    const WASM_PAGE_SIZE: u32 = 0x10000 /* 64kiB */;
    const MEMORY_RESERVATION: u32 = WASM_PAGE_SIZE * 16 * 64 /* 64MiB */;
    const MEMORY_GUARD_SIZE: u32 = WASM_PAGE_SIZE * 16 * 64 /* 64MiB */;
    const MEMORY_RESERVATION_FOR_GROWTH: u32 = WASM_PAGE_SIZE * 16 * 64 /* 64MiB */;

    let mut config = wasmtime::Config::new();
    config
        .cranelift_nan_canonicalization(true)
        .cranelift_opt_level(wasmtime::OptLevel::Speed)
        .memory_reservation(u64::from(MEMORY_RESERVATION))
        .memory_guard_size(u64::from(MEMORY_GUARD_SIZE))
        .memory_reservation_for_growth(u64::from(MEMORY_RESERVATION_FOR_GROWTH))
        .wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Enable)
        // WASM feature restrictions, follows the feature validation in
        //  numcodecs_wasm_host_reproducible::engine::ValidatedModule::new
        .wasm_bulk_memory(true)
        // wasmtime is compiled without the `component-model` feature
        // .wasm_component_model(false) and friends
        .wasm_custom_page_sizes(false)
        .wasm_exceptions(true)
        .wasm_extended_const(false)
        .wasm_function_references(false)
        .wasm_gc(false)
        .wasm_memory64(false)
        .wasm_multi_memory(true)
        .wasm_multi_value(true)
        .wasm_reference_types(true)
        .wasm_relaxed_simd(false)
        .relaxed_simd_deterministic(true)
        .wasm_simd(true)
        .wasm_shared_everything_threads(false)
        .wasm_stack_switching(false)
        .wasm_tail_call(false)
        // wasmtime is compiled without the `threads` feature
        // .wasm_threads(false)
        .wasm_wide_arithmetic(true);

    // TODO: allow configuration to be taken from somewhere else
    wasmtime::Cache::from_file(None)
        .map(|cache| config.cache(Some(cache)))
        .map_err(|err| PyErrChain::new(py, err))?;

    let engine = wasmtime::Engine::new(&config).map_err(|err| PyErrChain::new(py, err))?;

    Ok(Engine::new(engine))
}

#[cfg(target_arch = "wasm32")]
pub type Engine = pyodide_webassembly_runtime_layer::Engine;

#[cfg(target_arch = "wasm32")]
#[expect(clippy::unnecessary_wraps)]
pub fn default_engine(_py: Python) -> Result<Engine, PyErr> {
    Ok(Engine::default())
}
