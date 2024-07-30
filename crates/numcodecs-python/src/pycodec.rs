use std::{borrow::Cow, sync::Arc};

use numcodecs::{AnyCowArray, Codec, DynCodec};
use numpy::{ndarray::ArrayD, PyArray};
use pyo3::{
    buffer::PyBuffer,
    exceptions::{PyIndexError, PyTypeError},
    intern,
    prelude::*,
    types::{IntoPyDict, PyDict, PyDictMethods},
};
use pythonize::{Depythonizer, Pythonizer};
use serde::{Deserializer, Serializer};
use serde_transcode::transcode;

use crate::{CodecClassMethods, CodecMethods, Registry};

pub struct PyCodec {
    codec: Py<crate::Codec>,
    codec_id: Arc<String>,
}

impl Codec for PyCodec {
    type Error = PyErr;

    fn from_config<'de, D: Deserializer<'de>>(config: D) -> Result<Self, D::Error> {
        Python::with_gil(|py| {
            let config =
                transcode(config, Pythonizer::new(py)).map_err(serde::de::Error::custom)?;
            let config: Bound<PyDict> = config.extract(py).map_err(serde::de::Error::custom)?;

            let codec =
                Registry::get_codec(config.as_borrowed()).map_err(serde::de::Error::custom)?;
            let codec_id = codec.class().codec_id().map_err(serde::de::Error::custom)?;

            Ok(Self {
                codec: codec.unbind(),
                codec_id: Arc::new(codec_id),
            })
        })
    }

    fn encode<'a>(&self, data: AnyCowArray<'a>) -> Result<AnyCowArray<'a>, Self::Error> {
        Python::with_gil(|py| {
            let this = self.codec.bind(py).clone().into_any();
            #[allow(unsafe_code)] // FIXME
            let data = unsafe {
                match &data {
                    AnyCowArray::U8(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::U16(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::U32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::U64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::I8(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::I16(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::I32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::I64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::F32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyCowArray::F64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    _ => return Err(PyTypeError::new_err("unsupported type of data buffer")),
                }
            };
            // create a fully-immutable view of the data that is safe to pass to Python
            data.call_method(
                intern!(py, "setflags"),
                (),
                Some(&[(intern!(py, "write"), true)].into_py_dict_bound(py)),
            )?;
            let data = data.call_method0(intern!(py, "view"))?;

            let encoded = self.codec.bind(py).encode(data.as_borrowed())?;
            let encoded = if let Ok(e) = encoded.extract::<PyBuffer<u8>>() {
                AnyCowArray::U8(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<u16>>() {
                AnyCowArray::U16(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<u32>>() {
                AnyCowArray::U32(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<u64>>() {
                AnyCowArray::U64(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<i8>>() {
                AnyCowArray::I8(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<i16>>() {
                AnyCowArray::I16(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<i32>>() {
                AnyCowArray::I32(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<i64>>() {
                AnyCowArray::I64(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<f32>>() {
                AnyCowArray::F32(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = encoded.extract::<PyBuffer<f64>>() {
                AnyCowArray::F64(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else {
                return Err(PyTypeError::new_err("unsupported type of encoded buffer"));
            };

            Ok(encoded)
        })
    }

    fn decode<'a>(&self, encoded: AnyCowArray<'a>) -> Result<AnyCowArray<'a>, Self::Error> {
        Python::with_gil(|py| {
            let this = self.codec.bind(py).clone().into_any();
            #[allow(unsafe_code)] // FIXME
            let encoded = unsafe {
                match &encoded {
                    AnyCowArray::U8(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::U16(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::U32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::U64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::I8(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::I16(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::I32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::I64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::F32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyCowArray::F64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    _ => return Err(PyTypeError::new_err("unsupported type of encoded buffer")),
                }
            };
            // create a fully-immutable view of the data that is safe to pass to Python
            encoded.call_method(
                intern!(py, "setflags"),
                (),
                Some(&[(intern!(py, "write"), true)].into_py_dict_bound(py)),
            )?;
            let encoded = encoded.call_method0(intern!(py, "view"))?;

            // TODO: handle optional out parameter
            let decoded = self.codec.bind(py).decode(encoded.as_borrowed(), None)?;
            let decoded = if let Ok(e) = decoded.extract::<PyBuffer<u8>>() {
                AnyCowArray::U8(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<u16>>() {
                AnyCowArray::U16(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<u32>>() {
                AnyCowArray::U32(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<u64>>() {
                AnyCowArray::U64(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<i8>>() {
                AnyCowArray::I8(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<i16>>() {
                AnyCowArray::I16(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<i32>>() {
                AnyCowArray::I32(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<i64>>() {
                AnyCowArray::I64(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<f32>>() {
                AnyCowArray::F32(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else if let Ok(e) = decoded.extract::<PyBuffer<f64>>() {
                AnyCowArray::F64(
                    ArrayD::from_shape_vec(e.shape(), e.to_vec(py)?)
                        .map_err(|err| PyIndexError::new_err(format!("{err}")))?
                        .into(),
                )
            } else {
                return Err(PyTypeError::new_err("unsupported type of decoded buffer"));
            };

            Ok(decoded)
        })
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Python::with_gil(|py| {
            let config = self
                .codec
                .bind(py)
                .get_config()
                .map_err(serde::ser::Error::custom)?;

            transcode(
                &mut Depythonizer::from_object_bound(config.into_any()),
                serializer,
            )
        })
    }
}

impl DynCodec for PyCodec {
    fn codec_id(&self) -> Cow<str> {
        Cow::Borrowed(&self.codec_id)
    }
}

impl Clone for PyCodec {
    #[allow(clippy::expect_used)] // clone is *not* fallible
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let config = self
                .codec
                .bind(py)
                .get_config()
                .expect("getting codec config should not fail");

            // removing the `id` field may fail if the config doesn't contain it
            let _ = config.del_item(intern!(py, "id"));

            let codec = self
                .codec
                .bind(py)
                .class()
                .codec_from_config(config.as_borrowed())
                .expect("re-creating codec from config should not fail");

            Self {
                codec: codec.unbind(),
                codec_id: self.codec_id.clone(),
            }
        })
    }
}
