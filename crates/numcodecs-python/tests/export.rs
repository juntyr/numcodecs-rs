#![expect(missing_docs)]

use numcodecs::{
    AnyArray, AnyArrayBase, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodecType,
    StaticCodec, StaticCodecConfig, StaticCodecType,
};
use numcodecs_python::{
    export_codec_class, PyCodecAdapter, PyCodecClassAdapter, PyCodecClassMethods, PyCodecMethods,
    PyCodecRegistry,
};
use pyo3::{exceptions::PyTypeError, intern, prelude::*, types::PyDict};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use ::{
    convert_case as _, ndarray as _, pyo3_error as _, pythonize as _, serde as _, serde_json as _,
    serde_transcode as _, thiserror as _,
};

#[test]
fn export() -> Result<(), PyErr> {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "codecs")?;
        export_codec_class(
            py,
            StaticCodecType::<NegateCodec>::of(),
            module.as_borrowed(),
        )?;

        let config = PyDict::new(py);
        config.set_item("id", "negate.rs")?;

        // create a codec using registry lookup
        let codec = PyCodecRegistry::get_codec(config.as_borrowed())?;
        assert_eq!(codec.class().as_type().name()?.to_cow()?, "Negate");
        assert_eq!(codec.class().codec_id()?, "negate.rs");

        // check the codec's config
        let config = codec.get_config()?;
        assert_eq!(config.len(), 1);
        assert_eq!(
            config
                .get_item("id")?
                .map(|i| i.extract::<String>())
                .transpose()?
                .as_deref(),
            Some("negate.rs")
        );

        // encode and decode data with the codec
        let data = &[1.0_f64, 2.0, 3.0, 4.0];
        let encoded = codec.encode(numpy::PyArray1::from_slice(py, data).as_any().as_borrowed())?;
        let decoded = codec.decode(encoded.as_borrowed(), None)?;
        // decode into an output
        let decoded_out = numpy::PyArray1::<f64>::zeros(py, (4,), false);
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

#[test]
fn schema() -> Result<(), PyErr> {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "codecs")?;
        let class = export_codec_class(
            py,
            StaticCodecType::<NegateCodec>::of(),
            module.as_borrowed(),
        )?;

        let ty = PyCodecClassAdapter::from_codec_class(class.clone())?;
        assert_eq!(
            ty.codec_config_schema(),
            schema_for!(<NegateCodec as StaticCodec>::Config<'static>)
        );

        assert_eq!(
            class.getattr("__doc__")?.extract::<String>()?,
            "A codec that negates its inputs on encoding and decoding.

This codec does *not* take any parameters."
        );

        assert_eq!(
            format!(
                "{}",
                py.import(intern!(py, "inspect"))?
                    .getattr(intern!(py, "signature"))?
                    .call1((class.getattr(intern!(py, "__init__"))?,))?
            ),
            "(self)",
        );

        Ok(())
    })
}

#[test]
fn downcast() -> Result<(), PyErr> {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "codecs")?;
        let class = export_codec_class(
            py,
            StaticCodecType::<NegateCodec>::of(),
            module.as_borrowed(),
        )?;

        assert!(PyCodecClassAdapter::with_downcast(
            py,
            &class,
            |_: &StaticCodecType<NegateCodec>| ()
        )
        .is_some());

        let codec = class.codec_from_config(PyDict::new(py).as_borrowed())?;

        assert!(PyCodecAdapter::with_downcast(py, &codec, |_: &NegateCodec| ()).is_some());

        Ok(())
    })
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// A codec that negates its inputs on encoding and decoding.
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
}

impl StaticCodec for NegateCodec {
    const CODEC_ID: &'static str = "negate.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}
