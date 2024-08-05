use std::sync::Arc;

use numcodecs::{
    AnyArray, AnyArrayBase, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec,
    DynCodecType,
};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::{
    exceptions::PyTypeError,
    intern,
    prelude::*,
    types::{IntoPyDict, PyDict, PyDictMethods},
};
use pythonize::{Depythonizer, Pythonizer};
use serde::{Deserializer, Serializer};
use serde_transcode::transcode;

use crate::{CodecClassMethods, CodecMethods, Registry};

/// Wrapper around Python [`Codec`][`crate::Codec`]s to use the Rust [`Codec`]
/// API.
pub struct PyCodec {
    codec: Py<crate::Codec>,
    class: Py<crate::CodecClass>,
    codec_id: Arc<String>,
}

impl PyCodec {
    /// Instantiate a codec from the [`Registry`] with a serialized
    /// `config`uration.
    ///
    /// The config must include the `id` field with the
    /// [`PyCodecClass::codec_id`].
    ///
    /// # Errors
    ///
    /// Errors if no codec with a matching `id` has been registered, or if
    /// constructing the codec fails.
    pub fn from_registry_with_config<'de, D: Deserializer<'de>>(
        config: D,
    ) -> Result<Self, D::Error> {
        Python::with_gil(|py| {
            let config = transcode(config, Pythonizer::new(py))?;
            let config: Bound<PyDict> = config.extract(py)?;

            let codec = Registry::get_codec(config.as_borrowed())?;

            Self::from_codec(codec)
        })
        .map_err(serde::de::Error::custom)
    }

    /// Wraps a [`Codec`][`crate::Codec`] to use the Rust [`Codec`] API.
    ///
    /// # Errors
    ///
    /// Errors if the `codec`'s class does not provide an identifier.
    pub fn from_codec(codec: Bound<crate::Codec>) -> Result<Self, PyErr> {
        let class = codec.class();
        let codec_id = class.codec_id()?;

        Ok(Self {
            codec: codec.unbind(),
            class: class.unbind(),
            codec_id: Arc::new(codec_id),
        })
    }

    /// Access the wrapped [`Codec`][`crate::Codec`] to use its Python
    /// [`CodecMethods`] API.
    #[must_use]
    pub fn as_codec<'py>(&self, py: Python<'py>) -> &Bound<'py, crate::Codec> {
        self.codec.bind(py)
    }

    /// Unwrap the [`Codec`][`crate::Codec`] to use its Python [`CodecMethods`]
    /// API.
    #[must_use]
    pub fn into_codec(self, py: Python) -> Bound<crate::Codec> {
        self.codec.into_bound(py)
    }
}

impl Codec for PyCodec {
    type Error = PyErr;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Python::with_gil(|py| {
            let this = self.codec.bind(py).clone().into_any();
            #[allow(unsafe_code)] // FIXME
            let data = unsafe {
                match &data {
                    AnyArrayBase::U8(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::U16(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::U32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::U64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I8(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I16(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::F32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::F64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "unsupported type {} of data buffer",
                            data.dtype()
                        )))
                    }
                }
            };
            // create a fully-immutable view of the data that is safe to pass to Python
            data.call_method(
                intern!(py, "setflags"),
                (),
                Some(&[(intern!(py, "write"), false)].into_py_dict_bound(py)),
            )?;
            let data = data.call_method0(intern!(py, "view"))?;

            let encoded = self.codec.bind(py).encode(data.as_borrowed())?;
            let encoded: Bound<PyUntypedArray> = py
                .import_bound(intern!(py, "numpy"))?
                .getattr(intern!(py, "asarray"))?
                .call1((encoded,))?
                .extract()?;
            let encoded = if let Ok(e) = encoded.downcast::<PyArrayDyn<u8>>() {
                AnyArrayBase::U8(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<u16>>() {
                AnyArrayBase::U16(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<u32>>() {
                AnyArrayBase::U32(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<u64>>() {
                AnyArrayBase::U64(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<i8>>() {
                AnyArrayBase::I8(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<i16>>() {
                AnyArrayBase::I16(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<i32>>() {
                AnyArrayBase::I32(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<i64>>() {
                AnyArrayBase::I64(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<f32>>() {
                AnyArrayBase::F32(e.try_readonly()?.to_owned_array())
            } else if let Ok(e) = encoded.downcast::<PyArrayDyn<f64>>() {
                AnyArrayBase::F64(e.try_readonly()?.to_owned_array())
            } else {
                return Err(PyTypeError::new_err(format!(
                    "unsupported dtype {} of encoded buffer",
                    encoded.dtype()
                )));
            };

            Ok(encoded)
        })
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Python::with_gil(|py| {
            let this = self.codec.bind(py).clone().into_any();
            #[allow(unsafe_code)] // FIXME
            let encoded = unsafe {
                match &encoded {
                    AnyArrayBase::U8(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::U16(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::U32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::U64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I8(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I16(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::F32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::F64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "unsupported type {} of encoded buffer",
                            encoded.dtype()
                        )))
                    }
                }
            };
            // create a fully-immutable view of the data that is safe to pass to Python
            encoded.call_method(
                intern!(py, "setflags"),
                (),
                Some(&[(intern!(py, "write"), false)].into_py_dict_bound(py)),
            )?;
            let encoded = encoded.call_method0(intern!(py, "view"))?;

            let decoded = self.codec.bind(py).decode(encoded.as_borrowed(), None)?;
            let decoded: Bound<PyUntypedArray> = py
                .import_bound(intern!(py, "numpy"))?
                .getattr(intern!(py, "asarray"))?
                .call1((decoded,))?
                .extract()?;
            let decoded = if let Ok(d) = decoded.downcast::<PyArrayDyn<u8>>() {
                AnyArrayBase::U8(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<u16>>() {
                AnyArrayBase::U16(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<u32>>() {
                AnyArrayBase::U32(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<u64>>() {
                AnyArrayBase::U64(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<i8>>() {
                AnyArrayBase::I8(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<i16>>() {
                AnyArrayBase::I16(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<i32>>() {
                AnyArrayBase::I32(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<i64>>() {
                AnyArrayBase::I64(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<f32>>() {
                AnyArrayBase::F32(d.try_readonly()?.to_owned_array())
            } else if let Ok(d) = decoded.downcast::<PyArrayDyn<f64>>() {
                AnyArrayBase::F64(d.try_readonly()?.to_owned_array())
            } else {
                return Err(PyTypeError::new_err(format!(
                    "unsupported dtype {} of decoded buffer",
                    decoded.dtype()
                )));
            };

            Ok(decoded)
        })
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        Python::with_gil(|py| {
            let this = self.codec.bind(py).clone().into_any();
            #[allow(unsafe_code)] // FIXME
            let encoded = unsafe {
                match &encoded {
                    AnyArrayBase::U8(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::U16(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::U32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::U64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I8(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I16(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::I64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::F32(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    AnyArrayBase::F64(e) => PyArray::borrow_from_array_bound(e, this).into_any(),
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "unsupported type {} of encoded buffer",
                            encoded.dtype()
                        )))
                    }
                }
            };
            // create a fully-immutable view of the data that is safe to pass to Python
            encoded.call_method(
                intern!(py, "setflags"),
                (),
                Some(&[(intern!(py, "write"), false)].into_py_dict_bound(py)),
            )?;
            let encoded = encoded.call_method0(intern!(py, "view"))?;

            let this = self.codec.bind(py).clone().into_any();
            #[allow(unsafe_code)] // FIXME
            let decoded_in = unsafe {
                match &decoded {
                    AnyArrayBase::U8(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::U16(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::U32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::U64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I8(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I16(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::I64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::F32(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    AnyArrayBase::F64(d) => PyArray::borrow_from_array_bound(d, this).into_any(),
                    _ => {
                        return Err(PyTypeError::new_err(format!(
                            "unsupported type {} of decoded buffer",
                            decoded.dtype()
                        )))
                    }
                }
            };

            let decoded_out = self
                .codec
                .bind(py)
                .decode(encoded.as_borrowed(), Some(decoded_in.as_borrowed()))?;

            // Ideally, all codecs should just use the provided out array
            if decoded_out.is(&decoded_in) {
                return Ok(());
            }

            let decoded_out: Bound<PyUntypedArray> = py
                .import_bound(intern!(py, "numpy"))?
                .getattr(intern!(py, "asarray"))?
                .call1((decoded_out,))?
                .extract()?;
            if let Ok(d) = decoded_out.downcast::<PyArrayDyn<u8>>() {
                if let AnyArrayBase::U8(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<u16>>() {
                if let AnyArrayBase::U16(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<u32>>() {
                if let AnyArrayBase::U32(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<u64>>() {
                if let AnyArrayBase::U64(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<i8>>() {
                if let AnyArrayBase::I8(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<i16>>() {
                if let AnyArrayBase::I16(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<i32>>() {
                if let AnyArrayBase::I32(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<i64>>() {
                if let AnyArrayBase::I64(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<f32>>() {
                if let AnyArrayBase::F32(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else if let Ok(d) = decoded_out.downcast::<PyArrayDyn<f64>>() {
                if let AnyArrayBase::F64(mut decoded) = decoded {
                    return Ok(decoded.assign(&d.try_readonly()?.as_array()));
                }
            } else {
                return Err(PyTypeError::new_err(format!(
                    "unsupported dtype {} of decoded buffer",
                    decoded_out.dtype()
                )));
            };

            Err(PyTypeError::new_err(format!(
                "mismatching dtype {} of decoded buffer, expected {}",
                decoded_out.dtype(),
                decoded.dtype(),
            )))
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
                .class
                .bind(py)
                .codec_from_config(config.as_borrowed())
                .expect("re-creating codec from config should not fail");

            Self {
                codec: codec.unbind(),
                class: self.class.clone_ref(py),
                codec_id: self.codec_id.clone(),
            }
        })
    }
}

impl DynCodec for PyCodec {
    type Type = PyCodecClass;

    fn ty(&self) -> Self::Type {
        Python::with_gil(|py| PyCodecClass {
            class: self.class.clone_ref(py),
            codec_id: self.codec_id.clone(),
        })
    }
}

/// Wrapper around Python [`CodecClass`][`crate::CodecClass`]es to use the Rust
/// [`DynCodecType`] API.
pub struct PyCodecClass {
    class: Py<crate::CodecClass>,
    codec_id: Arc<String>,
}

impl PyCodecClass {
    /// Wraps a [`CodecClass`][`crate::CodecClass`] to use the Rust
    /// [`DynCodecType`] API.
    ///
    /// # Errors
    ///
    /// Errors if the codec `class` does not provide an identifier.
    pub fn from_codec_class(class: Bound<crate::CodecClass>) -> Result<Self, PyErr> {
        let codec_id = class.codec_id()?;

        Ok(Self {
            class: class.unbind(),
            codec_id: Arc::new(codec_id),
        })
    }

    /// Access the wrapped [`CodecClass`][`crate::CodecClass`] to use its Python
    /// [`CodecClassMethods`] API.
    #[must_use]
    pub fn as_codec_class<'py>(&self, py: Python<'py>) -> &Bound<'py, crate::CodecClass> {
        self.class.bind(py)
    }

    /// Unwrap the [`CodecClass`][`crate::CodecClass`] to use its Python
    /// [`CodecClassMethods`] API.
    #[must_use]
    pub fn into_codec_class(self, py: Python) -> Bound<crate::CodecClass> {
        self.class.into_bound(py)
    }
}

impl DynCodecType for PyCodecClass {
    type Codec = PyCodec;

    fn codec_id(&self) -> &str {
        &self.codec_id
    }

    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        Python::with_gil(|py| {
            let config =
                transcode(config, Pythonizer::new(py)).map_err(serde::de::Error::custom)?;
            let config: Bound<PyDict> = config.extract(py).map_err(serde::de::Error::custom)?;

            let codec = self
                .class
                .bind(py)
                .codec_from_config(config.as_borrowed())
                .map_err(serde::de::Error::custom)?;

            Ok(PyCodec {
                codec: codec.unbind(),
                class: self.class.clone_ref(py),
                codec_id: self.codec_id.clone(),
            })
        })
    }
}
