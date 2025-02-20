#![expect(missing_docs)]

use ndarray::{Array1, ArrayView1};
use numcodecs::{AnyArray, AnyArrayView, AnyCowArray, Codec, DynCodec, DynCodecType};
use numcodecs_python::{PyCodecAdapter, PyCodecClassMethods, PyCodecMethods, PyCodecRegistry};
use pyo3::{prelude::*, types::PyDict};
use pyo3_error::PyErrChain;
use serde_json::json;
use ::{
    convert_case as _, pythonize as _, schemars as _, serde as _, serde_transcode as _,
    thiserror as _,
};

#[test]
fn python_api() -> Result<(), PyErr> {
    Python::with_gil(|py| {
        let config = PyDict::new(py);
        config.set_item("id", "crc32")?;

        // create a codec using registry lookup
        let codec = PyCodecRegistry::get_codec(config.as_borrowed())?;
        assert_eq!(codec.class().codec_id()?, "crc32");

        // re-register the codec class under a custom name
        let class = codec.class();
        PyCodecRegistry::register_codec(class.as_borrowed(), Some("my-crc32"))?;
        config.set_item("id", "my-crc32")?;

        // create a codec using registry lookup of the custom name
        let codec = PyCodecRegistry::get_codec(config.as_borrowed())?;
        assert_eq!(codec.class().codec_id()?, "crc32");

        // create a codec using the class
        let codec = class.codec_from_config(PyDict::new(py).as_borrowed())?;

        // check the codec's config
        let config = codec.get_config()?;
        assert_eq!(config.len(), 1);
        assert_eq!(
            config
                .get_item("id")?
                .map(|i| i.extract::<String>())
                .transpose()?
                .as_deref(),
            Some("crc32")
        );

        // encode and decode data with the codec
        let data = &[1_u8, 2, 3, 4];
        let encoded = codec.encode(numpy::PyArray1::from_slice(py, data).as_any().as_borrowed())?;
        let decoded = codec.decode(encoded.as_borrowed(), None)?;
        // decode into an output
        let decoded_out = numpy::PyArray1::<u8>::zeros(py, (4,), false);
        codec.decode(
            encoded.as_borrowed(),
            Some(decoded_out.as_any().as_borrowed()),
        )?;

        // check the encoded and decoded data
        let encoded: Vec<u8> = encoded.extract()?;
        let decoded: Vec<u8> = decoded.extract()?;
        let decoded_out: Vec<u8> = decoded_out.extract()?;
        assert_eq!(encoded, [205, 251, 60, 182, 1, 2, 3, 4]);
        assert_eq!(decoded, data);
        assert_eq!(decoded_out, data);

        Ok(())
    })
}

#[test]
fn rust_api() -> Result<(), PyErr> {
    // create a codec using registry lookup
    let codec = PyCodecAdapter::from_registry_with_config(json!({
        "id": "crc32",
    }))
    .map_err(|err| Python::with_gil(|py| PyErrChain::pyerr_from_err(py, err)))?;
    assert_eq!(codec.ty().codec_id(), "crc32");

    // clone the codec
    #[expect(clippy::redundant_clone)]
    let codec = codec.clone();

    // create a codec using the type object
    let codec = codec
        .ty()
        .codec_from_config(json!({}))
        .map_err(|err| Python::with_gil(|py| PyErrChain::pyerr_from_err(py, err)))?;

    // check the codec's config
    let config = codec
        .get_config(serde_json::value::Serializer)
        .map_err(|err| Python::with_gil(|py| PyErrChain::pyerr_from_err(py, err)))?;
    assert_eq!(
        config,
        json!({
            "id": "crc32",
        })
    );

    // encode and decode data with the codec
    let data = &[1_u8, 2, 3, 4];
    let encoded = codec.encode(AnyCowArray::U8(ArrayView1::from(data).into_dyn().into()))?;
    let decoded = codec.decode(encoded.cow())?;
    // decode into an output
    let mut decoded_into = AnyArray::U8(Array1::zeros(4).into_dyn());
    codec.decode_into(encoded.view(), decoded_into.view_mut())?;

    // check the encoded and decoded data
    assert_eq!(
        encoded.view(),
        AnyArrayView::U8(ArrayView1::from(&[205, 251, 60, 182, 1, 2, 3, 4]).into_dyn())
    );
    assert_eq!(
        decoded.view(),
        AnyArrayView::U8(ArrayView1::from(data).into_dyn())
    );
    assert_eq!(
        decoded_into.view(),
        AnyArrayView::U8(ArrayView1::from(data).into_dyn())
    );

    Ok(())
}
