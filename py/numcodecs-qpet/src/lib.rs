use numcodecs::StaticCodecType;
use numcodecs_qpet::SperrCodec;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_qpet")]
fn qpet<'py>(py: Python<'py>, module: &Bound<'py, PyModule>) -> Result<(), PyErr> {
    numcodecs_python::export_codec_class(
        py,
        StaticCodecType::<SperrCodec>::of(),
        module.as_borrowed(),
    )?;

    Ok(())
}
