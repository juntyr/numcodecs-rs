use numcodecs::{
    AnyArray, AnyArrayBase, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec,
    DynCodecType, StaticCodec, StaticCodecType,
};
use numcodecs_python::{export_codec_class, CodecClassMethods, CodecMethods, PyCodec, Registry};
use numpy::ndarray::{Array1, ArrayView1};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::PyDict,
};
use serde::ser::SerializeMap;
use serde_json::json;
use ::{convert_case as _, pythonize as _, serde as _, serde_transcode as _};

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
        // decode into an output
        let decoded_out = numpy::PyArray1::<u8>::zeros_bound(py, (4,), false);
        codec.decode(encoded.as_borrowed(), Some(decoded.as_any().as_borrowed()))?;

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
    let codec = PyCodec::from_registry_with_config(json!({
        "id": "crc32",
    }))
    .map_err(|err| PyRuntimeError::new_err(format!("{err}")))?;
    assert_eq!(codec.ty().codec_id(), "crc32");

    // clone the codec
    let codec = codec.clone();

    // create a codec using the type object
    let codec = codec
        .ty()
        .codec_from_config(json!({}))
        .map_err(|err| PyRuntimeError::new_err(format!("{err}")))?;

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

#[test]
fn export() -> Result<(), PyErr> {
    #[derive(Copy, Clone)]
    struct NegateCodec;

    impl Codec for NegateCodec {
        type Error = PyErr;

        fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
            match data {
                AnyArrayBase::F64(a) => Ok(AnyArrayBase::F64(a.map(|x| -x))),
                _ => Err(PyTypeError::new_err("negate only supports f64")),
            }
        }

        fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
            match encoded {
                AnyArrayBase::F64(a) => Ok(AnyArrayBase::F64(a.map(|x| -x))),
                _ => Err(PyTypeError::new_err("negate only supports f64")),
            }
        }

        fn decode_into(
            &self,
            encoded: AnyArrayView,
            decoded: AnyArrayViewMut,
        ) -> Result<(), Self::Error> {
            match (encoded, decoded) {
                (AnyArrayBase::F64(e), AnyArrayBase::F64(mut d)) => {
                    d.assign(&e);
                    d.map_inplace(|x| *x = -(*x));
                    Ok(())
                }
                _ => Err(PyTypeError::new_err("negate only supports f64")),
            }
        }

        fn get_config<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(None)?;
            map.serialize_entry("id", Self::CODEC_ID)?;
            map.end()
        }
    }

    impl StaticCodec for NegateCodec {
        const CODEC_ID: &'static str = "negate";

        fn from_config<'de, D: serde::Deserializer<'de>>(_config: D) -> Result<Self, D::Error> {
            Ok(Self)
        }
    }

    Python::with_gil(|py| {
        let module = PyModule::new_bound(py, "codecs")?;
        export_codec_class(
            py,
            StaticCodecType::<NegateCodec>::of(),
            module.as_borrowed(),
        )?;

        let config = PyDict::new_bound(py);
        config.set_item("id", "negate")?;

        // create a codec using registry lookup
        let codec = Registry::get_codec(config.as_borrowed())?;
        assert_eq!(codec.class().codec_id()?, "negate");

        // check the codec's config
        let config = codec.get_config()?;
        assert_eq!(config.len(), 1);
        assert_eq!(
            config
                .get_item("id")?
                .map(|i| i.extract::<String>())
                .transpose()?
                .as_deref(),
            Some("negate")
        );

        // encode and decode data with the codec
        let data = &[1.0_f64, 2.0, 3.0, 4.0];
        let encoded = codec.encode(
            numpy::PyArray1::from_slice_bound(py, data)
                .as_any()
                .as_borrowed(),
        )?;
        let decoded = codec.decode(encoded.as_borrowed(), None)?;
        // decode into an output
        let decoded_out = numpy::PyArray1::<f64>::zeros_bound(py, (4,), false);
        codec.decode(encoded.as_borrowed(), Some(decoded.as_any().as_borrowed()))?;

        // check the encoded and decoded data
        let encoded: Vec<f64> = encoded.extract()?;
        let decoded: Vec<f64> = decoded.extract()?;
        let decoded_out: Vec<f64> = decoded_out.extract()?;
        assert_eq!(encoded, [-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(decoded, data);
        assert_eq!(decoded_out, data);

        Ok(())
    })
}
