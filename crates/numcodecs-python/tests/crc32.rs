use numcodecs::{AnyCowArray, Codec, DynCodec};
use numcodecs_python::{CodecClassMethods, CodecMethods, PyCodec, Registry};
use numpy::ndarray::ArrayView1;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};
use serde_json::json;
use ::{pythonize as _, serde as _, serde_transcode as _};

#[test]
fn python_api() -> Result<(), PyErr> {
    Python::with_gil(|py| {
        let config = PyDict::new_bound(py);
        config.set_item("id", "crc32")?;

        // create a codec using registry lookup
        let codec = Registry::get_codec(config.as_borrowed())?;
        assert_eq!(codec.class().codec_id()?, "crc32");

        // re-register the codec class under a custom name
        let class = codec.class();
        Registry::register_codec(class.as_borrowed(), Some("my-crc32"))?;
        config.set_item("id", "my-crc32")?;

        // create a codec using registry lookup of the custom name
        let codec = Registry::get_codec(config.as_borrowed())?;
        assert_eq!(codec.class().codec_id()?, "crc32");

        // create a codec using the class
        let codec = class.codec_from_config(PyDict::new_bound(py).as_borrowed())?;

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
        let encoded = codec.encode(
            numpy::PyArray1::from_slice_bound(py, data)
                .as_any()
                .as_borrowed(),
        )?;
        let decoded = codec.decode(encoded.as_borrowed(), None)?;

        // check the encoded and decoded data
        let encoded: Vec<u8> = encoded.extract()?;
        let decoded: Vec<u8> = decoded.extract()?;
        assert_eq!(encoded, [205, 251, 60, 182, 1, 2, 3, 4]);
        assert_eq!(decoded, data);

        Ok(())
    })
}

#[test]
fn rust_api() -> Result<(), PyErr> {
    // create a codec using registry lookup
    let codec = PyCodec::from_config(json!({
        "id": "crc32",
    }))
    .map_err(|err| PyRuntimeError::new_err(format!("{err}")))?;
    assert_eq!(codec.codec_id(), "crc32");

    // clone the codec
    let codec = codec.clone();

    // check the codec's config
    let config = codec
        .get_config(serde_json::value::Serializer)
        .map_err(|err| PyRuntimeError::new_err(format!("{err}")))?;
    assert_eq!(
        config,
        json!({
            "id": "crc32",
        })
    );

    // encode and decode data with the codec
    let data = &[1_u8, 2, 3, 4];
    let encoded = codec.encode(AnyCowArray::U8(ArrayView1::from(data).into_dyn().into()))?;
    let decoded = codec.decode(encoded.clone())?;

    // check the encoded and decoded data
    assert_eq!(
        encoded,
        AnyCowArray::U8(
            ArrayView1::from(&[205, 251, 60, 182, 1, 2, 3, 4])
                .into_dyn()
                .into()
        )
    );
    assert_eq!(
        decoded,
        AnyCowArray::U8(ArrayView1::from(data).into_dyn().into())
    );

    Ok(())
}
