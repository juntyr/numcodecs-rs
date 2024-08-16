use numcodecs::{
    serialize_codec_config_with_id, AnyArray, AnyArrayBase, AnyArrayView, AnyArrayViewMut,
    AnyCowArray, Codec, StaticCodec, StaticCodecType,
};
use numcodecs_python::{export_codec_class, PyCodecClassMethods, PyCodecMethods, PyCodecRegistry};
use pyo3::{exceptions::PyTypeError, prelude::*, types::PyDict};
use serde::{Deserialize, Serialize};
use ::{
    convert_case as _, ndarray as _, pythonize as _, serde as _, serde_json as _,
    serde_transcode as _,
};

#[test]
fn export() -> Result<(), PyErr> {
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
        let codec = PyCodecRegistry::get_codec(config.as_borrowed())?;
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
        codec.decode(
            encoded.as_borrowed(),
            Some(decoded_out.as_any().as_borrowed()),
        )?;

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

#[derive(Copy, Clone, Serialize, Deserialize)]
struct NegateCodec {
    // empty
}

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
        serialize_codec_config_with_id(self, self, serializer)
    }
}

impl StaticCodec for NegateCodec {
    const CODEC_ID: &'static str = "negate";

    fn from_config<'de, D: serde::Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Self::deserialize(config)
    }
}
