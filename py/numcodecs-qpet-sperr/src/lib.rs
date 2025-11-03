use numcodecs::StaticCodecType;
use numcodecs_qpet_sperr::QpetSperrCodec;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_qpet_sperr")]
fn qpet_sperr<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> Result<(), PyErr> {
    numcodecs_python::export_codec_class(
        py,
        StaticCodecType::<QpetSperrCodec>::of(),
        module.as_borrowed(),
    )?;

    Ok(())
}
