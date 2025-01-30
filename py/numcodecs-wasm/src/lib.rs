use numcodecs_python::PyCodecClass;
use numcodecs_wasm_host_reproducible::ReproducibleWasmCodecType;
use pyo3::prelude::*;

mod engine;

#[pymodule]
#[pyo3(name = "_wasm")]
fn wasm<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> Result<(), PyErr> {
    let logger = pyo3_log::Logger::new(py, pyo3_log::Caching::Nothing)?;
    logger
        .install()
        .map_err(|err| pyo3_error::PyErrChain::new(py, err))?;

    module.add_function(wrap_pyfunction!(create_codec_class, module)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(name = "_create_codec_class")]
fn create_codec_class<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyModule>,
    wasm: Vec<u8>,
) -> Result<Bound<'py, PyCodecClass>, PyErr> {
    let engine = engine::default_engine(py)?;

    let codec_ty = ReproducibleWasmCodecType::new(engine, wasm)
        .map_err(|err| pyo3_error::PyErrChain::new(py, err))?;

    let codec_class = numcodecs_python::export_codec_class(py, codec_ty, module.as_borrowed())?;

    Ok(codec_class)
}
