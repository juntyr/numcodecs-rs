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

use crate::{PyCodec, PyCodecClass, PyCodecClassMethods, PyCodecMethods, PyCodecRegistry};

/// Wrapper around [`PyCodec`]s to use the [`Codec`] API.
pub struct PyCodecAdapter {
    codec: Py<PyCodec>,
    class: Py<PyCodecClass>,
    codec_id: Arc<String>,
}

impl PyCodecAdapter {
    /// Instantiate a codec from the [`PyCodecRegistry`] with a serialized
    /// `config`uration.
    ///
    /// The config must include the `id` field with the
    /// [`PyCodecClassMethods::codec_id`].
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

            let codec = PyCodecRegistry::get_codec(config.as_borrowed())?;

            Self::from_codec(codec)
        })
        .map_err(serde::de::Error::custom)
    }

    /// Wraps a [`PyCodec`] to use the [`Codec`] API.
    ///
    /// # Errors
    ///
    /// Errors if the `codec`'s class does not provide an identifier.
    pub fn from_codec(codec: Bound<PyCodec>) -> Result<Self, PyErr> {
        let class = codec.class();
        let codec_id = class.codec_id()?;

        Ok(Self {
            codec: codec.unbind(),
            class: class.unbind(),
            codec_id: Arc::new(codec_id),
        })
    }

    /// Access the wrapped [`PyCodec`] to use its [`PyCodecMethods`] API.
    #[must_use]
    pub fn as_codec<'py>(&self, py: Python<'py>) -> &Bound<'py, PyCodec> {
        self.codec.bind(py)
    }

    /// Unwrap the [`PyCodec`] to use its [`PyCodecMethods`] API.
    #[must_use]
    pub fn into_codec(self, py: Python) -> Bound<PyCodec> {
        self.codec.into_bound(py)
    }

    /// Try to [`clone`][`Clone::clone`] this codec.
    ///
    /// # Errors
    ///
    /// Errors if extracting this codec's config or creating a new codec from
    /// the config fails.
    pub fn try_clone(&self, py: Python) -> Result<Self, PyErr> {
        let config = self.codec.bind(py).get_config()?;

        // removing the `id` field may fail if the config doesn't contain it
        let _ = config.del_item(intern!(py, "id"));

        let codec = self
            .class
            .bind(py)
            .codec_from_config(config.as_borrowed())?;

        Ok(Self {
            codec: codec.unbind(),
            class: self.class.clone_ref(py),
            codec_id: self.codec_id.clone(),
        })
    }
}

impl Codec for PyCodecAdapter {
    type Error = PyErr;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Python::with_gil(|py| {
            self.with_any_array_view_as_ndarray(py, &data.view(), |data| {
                let encoded = self.codec.bind(py).encode(data.as_borrowed())?;

                Self::any_array_from_ndarray_like(py, encoded)
            })
        })
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Python::with_gil(|py| {
            self.with_any_array_view_as_ndarray(py, &encoded.view(), |encoded| {
                let decoded = self.codec.bind(py).decode(encoded.as_borrowed(), None)?;

                Self::any_array_from_ndarray_like(py, decoded)
            })
        })
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        mut decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        Python::with_gil(|py| {
            let decoded_out = self.with_any_array_view_as_ndarray(py, &encoded, |encoded| {
                self.with_any_array_view_mut_as_ndarray(py, &mut decoded, |decoded_in| {
                    let decoded_out = self
                        .codec
                        .bind(py)
                        .decode(encoded.as_borrowed(), Some(decoded_in.as_borrowed()))?;

                    // Ideally, all codecs should just use the provided out array
                    if decoded_out.is(decoded_in) {
                        Ok(Ok(()))
                    } else {
                        Ok(Err(decoded_out.unbind()))
                    }
                })
            })?;
            let decoded_out = match decoded_out {
                Ok(()) => return Ok(()),
                Err(decoded_out) => decoded_out.into_bound(py),
            };

            // Otherwise, we force-copy the output into the decoded array
            Self::copy_into_any_array_view_mut_from_ndarray_like(py, &mut decoded, decoded_out)
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

impl PyCodecAdapter {
    fn with_any_array_view_as_ndarray<T>(
        &self,
        py: Python,
        view: &AnyArrayView,
        with: impl for<'a> FnOnce(&'a Bound<PyAny>) -> Result<T, PyErr>,
    ) -> Result<T, PyErr> {
        let this = self.codec.bind(py).clone().into_any();

        #[allow(unsafe_code)] // FIXME: we trust Python code to not store this array
        let ndarray = unsafe {
            match &view {
                AnyArrayBase::U8(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::U16(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::U32(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::U64(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I8(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I16(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I32(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I64(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::F32(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::F64(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "unsupported type {} of read-only array view",
                        view.dtype()
                    )))
                }
            }
        };

        // create a fully-immutable view of the data that is safe to pass to Python
        ndarray.call_method(
            intern!(py, "setflags"),
            (),
            Some(&[(intern!(py, "write"), false)].into_py_dict_bound(py)),
        )?;
        let view = ndarray.call_method0(intern!(py, "view"))?;

        with(&view)
    }

    fn with_any_array_view_mut_as_ndarray<T>(
        &self,
        py: Python,
        view_mut: &mut AnyArrayViewMut,
        with: impl for<'a> FnOnce(&'a Bound<PyAny>) -> Result<T, PyErr>,
    ) -> Result<T, PyErr> {
        let this = self.codec.bind(py).clone().into_any();

        #[allow(unsafe_code)] // FIXME: we trust Python code to not store this array
        let ndarray = unsafe {
            match &view_mut {
                AnyArrayBase::U8(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::U16(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::U32(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::U64(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I8(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I16(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I32(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::I64(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::F32(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                AnyArrayBase::F64(v) => PyArray::borrow_from_array_bound(v, this).into_any(),
                _ => {
                    return Err(PyTypeError::new_err(format!(
                        "unsupported type {} of read-only array view",
                        view_mut.dtype()
                    )))
                }
            }
        };

        with(&ndarray)
    }

    fn any_array_from_ndarray_like(
        py: Python,
        array_like: Bound<PyAny>,
    ) -> Result<AnyArray, PyErr> {
        let ndarray: Bound<PyUntypedArray> = py
            .import_bound(intern!(py, "numpy"))?
            .getattr(intern!(py, "asarray"))?
            .call1((array_like,))?
            .extract()?;

        let array = if let Ok(e) = ndarray.downcast::<PyArrayDyn<u8>>() {
            AnyArrayBase::U8(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<u16>>() {
            AnyArrayBase::U16(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<u32>>() {
            AnyArrayBase::U32(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<u64>>() {
            AnyArrayBase::U64(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<i8>>() {
            AnyArrayBase::I8(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<i16>>() {
            AnyArrayBase::I16(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<i32>>() {
            AnyArrayBase::I32(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<i64>>() {
            AnyArrayBase::I64(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<f32>>() {
            AnyArrayBase::F32(e.try_readonly()?.to_owned_array())
        } else if let Ok(e) = ndarray.downcast::<PyArrayDyn<f64>>() {
            AnyArrayBase::F64(e.try_readonly()?.to_owned_array())
        } else {
            return Err(PyTypeError::new_err(format!(
                "unsupported dtype {} of array-like",
                ndarray.dtype()
            )));
        };

        Ok(array)
    }

    fn copy_into_any_array_view_mut_from_ndarray_like(
        py: Python,
        view_mut: &mut AnyArrayViewMut,
        array_like: Bound<PyAny>,
    ) -> Result<(), PyErr> {
        let ndarray: Bound<PyUntypedArray> = py
            .import_bound(intern!(py, "numpy"))?
            .getattr(intern!(py, "asarray"))?
            .call1((array_like,))?
            .extract()?;

        #[allow(clippy::unit_arg)]
        if let Ok(d) = ndarray.downcast::<PyArrayDyn<u8>>() {
            if let AnyArrayBase::U8(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<u16>>() {
            if let AnyArrayBase::U16(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<u32>>() {
            if let AnyArrayBase::U32(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<u64>>() {
            if let AnyArrayBase::U64(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i8>>() {
            if let AnyArrayBase::I8(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i16>>() {
            if let AnyArrayBase::I16(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i32>>() {
            if let AnyArrayBase::I32(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i64>>() {
            if let AnyArrayBase::I64(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<f32>>() {
            if let AnyArrayBase::F32(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<f64>>() {
            if let AnyArrayBase::F64(ref mut view_mut) = view_mut {
                return Ok(view_mut.assign(&d.try_readonly()?.as_array()));
            }
        } else {
            return Err(PyTypeError::new_err(format!(
                "unsupported dtype {} of array-like",
                ndarray.dtype()
            )));
        };

        Err(PyTypeError::new_err(format!(
            "mismatching dtype {} of array-like, expected {}",
            ndarray.dtype(),
            view_mut.dtype(),
        )))
    }
}

impl Clone for PyCodecAdapter {
    fn clone(&self) -> Self {
        #[allow(clippy::expect_used)] // clone is *not* fallible
        Python::with_gil(|py| {
            self.try_clone(py)
                .expect("cloning a PyCodec should not fail")
        })
    }
}

impl DynCodec for PyCodecAdapter {
    type Type = PyCodecClassAdapter;

    fn ty(&self) -> Self::Type {
        Python::with_gil(|py| PyCodecClassAdapter {
            class: self.class.clone_ref(py),
            codec_id: self.codec_id.clone(),
        })
    }
}

/// Wrapper around [`PyCodecClass`]es to use the [`DynCodecType`] API.
pub struct PyCodecClassAdapter {
    class: Py<PyCodecClass>,
    codec_id: Arc<String>,
}

impl PyCodecClassAdapter {
    /// Wraps a [`PyCodecClass`] to use the [`DynCodecType`] API.
    ///
    /// # Errors
    ///
    /// Errors if the codec `class` does not provide an identifier.
    pub fn from_codec_class(class: Bound<PyCodecClass>) -> Result<Self, PyErr> {
        let codec_id = class.codec_id()?;

        Ok(Self {
            class: class.unbind(),
            codec_id: Arc::new(codec_id),
        })
    }

    /// Access the wrapped [`PyCodecClass`] to use its [`PyCodecClassMethods`]
    /// API.
    #[must_use]
    pub fn as_codec_class<'py>(&self, py: Python<'py>) -> &Bound<'py, PyCodecClass> {
        self.class.bind(py)
    }

    /// Unwrap the [`PyCodecClass`] to use its [`PyCodecClassMethods`] API.
    #[must_use]
    pub fn into_codec_class(self, py: Python) -> Bound<PyCodecClass> {
        self.class.into_bound(py)
    }
}

impl DynCodecType for PyCodecClassAdapter {
    type Codec = PyCodecAdapter;

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

            Ok(PyCodecAdapter {
                codec: codec.unbind(),
                class: self.class.clone_ref(py),
                codec_id: self.codec_id.clone(),
            })
        })
    }
}
