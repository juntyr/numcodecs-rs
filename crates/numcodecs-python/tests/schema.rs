#![expect(missing_docs)]

use numcodecs::DynCodecType;
use numcodecs_python::{PyCodecClass, PyCodecClassAdapter};
use pyo3::{intern, prelude::*};
use ::{
    convert_case as _, ndarray as _, numpy as _, pyo3_error as _, pythonize as _, schemars as _,
    serde as _, serde_json as _, serde_transcode as _, thiserror as _,
};

#[test]
fn collect_schemas() -> Result<(), PyErr> {
    Python::with_gil(|py| {
        let registry = py
            .import(intern!(py, "numcodecs"))?
            .getattr(intern!(py, "registry"))?
            .getattr(intern!(py, "codec_registry"))?;

        for codec in registry.call_method0(intern!(py, "items"))?.try_iter()? {
            let (codec_id, codec_class): (String, Bound<PyCodecClass>) = codec?.extract()?;

            let codec_ty = PyCodecClassAdapter::from_codec_class(codec_class)?;

            println!(
                "{codec_id}: {:#}",
                codec_ty.codec_config_schema().as_value()
            );
        }

        Ok(())
    })
}
