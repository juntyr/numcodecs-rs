use numcodecs::StaticCodecType;
use pyo3::prelude::*;
use numcodecs_my_codec::%codec-path% as MyCodec;

#[pymodule]
#[pyo3(name = "_native")]
fn native<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> Result<(), PyErr> {
    let logger = pyo3_log::Logger::new(py, pyo3_log::Caching::Nothing)?;
    logger
        .install()
        .map_err(|err| pyo3_error::PyErrChain::new(py, err))?;

    numcodecs_python::export_codec_class(
        py, StaticCodecType::<MyCodec>::of(), module.as_borrowed()
    )?;

    Ok(())
}
