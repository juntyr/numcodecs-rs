use numcodecs_python::{PyCodec, PyCodecAdapter, PyCodecClass};
use numcodecs_wasm_host_reproducible::{ReproducibleWasmCodec, ReproducibleWasmCodecType};
use pyo3::{exceptions::PyTypeError, prelude::*};

mod engine;

use engine::{default_engine, Engine};

#[pymodule]
#[pyo3(name = "_wasm")]
fn wasm<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> Result<(), PyErr> {
    let logger = pyo3_log::Logger::new(py, pyo3_log::Caching::Nothing)?;
    logger
        .install()
        .map_err(|err| pyo3_error::PyErrChain::new(py, err))?;

    module.add_function(wrap_pyfunction!(create_codec_class, module)?)?;
    module.add_function(wrap_pyfunction!(read_codec_instruction_counter, module)?)?;

    Ok(())
}

#[pyfunction]
#[pyo3(name = "_create_codec_class")]
fn create_codec_class<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyModule>,
    wasm: Vec<u8>,
) -> Result<Bound<'py, PyCodecClass>, PyErr> {
    let engine = default_engine(py)?;

    let codec_ty = ReproducibleWasmCodecType::new(engine, wasm)
        .map_err(|err| pyo3_error::PyErrChain::new(py, err))?;

    let codec_class = numcodecs_python::export_codec_class(py, codec_ty, module.as_borrowed())?;

    Ok(codec_class)
}

#[pyfunction]
#[pyo3(name = "_read_codec_instruction_counter")]
fn read_codec_instruction_counter<'py>(
    py: Python<'py>,
    codec: &Bound<'py, PyCodec>,
) -> Result<u64, PyErr> {
    let Some(instruction_counter) =
        PyCodecAdapter::with_downcast(py, codec, |codec: &ReproducibleWasmCodec<Engine>| {
            codec.instruction_counter()
        })
        .transpose()
        .map_err(|err| pyo3_error::PyErrChain::new(py, err))?
    else {
        return Err(PyTypeError::new_err(
            "`codec` is not a wasm codec, only wasm codecs have instruction counts",
        ));
    };

    Ok(instruction_counter.0)
}
