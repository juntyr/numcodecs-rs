use std::sync::Arc;

use ndarray::{ArrayBase, DataMut, Dimension};
use numcodecs::{
    AnyArray, AnyArrayBase, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec,
    DynCodecType,
};
use numpy::{Element, PyArray, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    marker::Ungil,
    prelude::*,
    types::{IntoPyDict, PyDict, PyDictMethods},
};
use pythonize::{Depythonizer, Pythonizer};
use schemars::Schema;
use serde::{Deserializer, Serializer};
use serde_transcode::transcode;

use crate::{
    export::{RustCodec, RustCodecType},
    schema::schema_from_codec_class,
    utils::numpy_asarray,
    PyCodec, PyCodecClass, PyCodecClassMethods, PyCodecMethods, PyCodecRegistry,
};

/// Wrapper around [`PyCodec`]s to use the [`Codec`] API.
pub struct PyCodecAdapter {
    codec: Py<PyCodec>,
    class: Py<PyCodecClass>,
    codec_id: Arc<String>,
    codec_config_schema: Arc<Schema>,
}

impl PyCodecAdapter {
    /// Instantiate a codec from the [`PyCodecRegistry`] with a serialized
    /// `config`uration.
    ///
    /// The config *must* include the `id` field with the
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
            let config: Bound<PyDict> = config.extract()?;

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
        let codec_config_schema = schema_from_codec_class(class.py(), &class).map_err(|err| {
            PyTypeError::new_err(format!(
                "failed to extract the {codec_id} codec config schema: {err}"
            ))
        })?;

        Ok(Self {
            codec: codec.unbind(),
            class: class.unbind(),
            codec_id: Arc::new(codec_id),
            codec_config_schema: Arc::new(codec_config_schema),
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
            codec_config_schema: self.codec_config_schema.clone(),
        })
    }

    /// If `codec` represents an exported [`DynCodec`] `T`, i.e. its class was
    /// initially created with [`crate::export_codec_class`], the `with` closure
    /// provides access to the instance of type `T`.
    ///
    /// If `codec` is not an instance of `T`, the `with` closure is *not* run
    /// and `None` is returned.
    pub fn with_downcast<T: DynCodec, O: Ungil>(
        py: Python,
        codec: &Bound<PyCodec>,
        with: impl Send + Ungil + for<'a> FnOnce(&'a T) -> O,
    ) -> Option<O> {
        let Ok(codec) = codec.downcast::<RustCodec>() else {
            return None;
        };

        let codec = codec.get().downcast()?;

        // The `with` closure contains arbitrary Rust code and may block,
        // which we cannot allow while holding the GIL
        Some(py.allow_threads(|| with(codec)))
    }
}

impl Codec for PyCodecAdapter {
    type Error = PyErr;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Python::with_gil(|py| {
            self.with_any_array_view_as_ndarray(py, &data.view(), |data| {
                let encoded = self.codec.bind(py).encode(data.as_borrowed())?;

                Self::any_array_from_ndarray_like(py, encoded.as_borrowed())
            })
        })
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Python::with_gil(|py| {
            self.with_any_array_view_as_ndarray(py, &encoded.view(), |encoded| {
                let decoded = self.codec.bind(py).decode(encoded.as_borrowed(), None)?;

                Self::any_array_from_ndarray_like(py, decoded.as_borrowed())
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
            Self::copy_into_any_array_view_mut_from_ndarray_like(
                py,
                &mut decoded,
                decoded_out.as_borrowed(),
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

        #[expect(unsafe_code)] // FIXME: we trust Python code to not store this array
        let ndarray = unsafe {
            match &view {
                AnyArrayBase::U8(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::U16(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::U32(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::U64(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I8(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I16(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I32(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I64(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::F32(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::F64(v) => PyArray::borrow_from_array(v, this).into_any(),
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
            Some(&[(intern!(py, "write"), false)].into_py_dict(py)?),
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

        #[expect(unsafe_code)] // FIXME: we trust Python code to not store this array
        let ndarray = unsafe {
            match &view_mut {
                AnyArrayBase::U8(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::U16(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::U32(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::U64(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I8(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I16(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I32(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::I64(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::F32(v) => PyArray::borrow_from_array(v, this).into_any(),
                AnyArrayBase::F64(v) => PyArray::borrow_from_array(v, this).into_any(),
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
        array_like: Borrowed<PyAny>,
    ) -> Result<AnyArray, PyErr> {
        let ndarray = numpy_asarray(py, array_like)?;

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
        array_like: Borrowed<PyAny>,
    ) -> Result<(), PyErr> {
        fn shape_checked_assign<
            T: Copy + Element,
            S2: DataMut<Elem = T>,
            D1: Dimension,
            D2: Dimension,
        >(
            src: &Bound<PyArray<T, D1>>,
            dst: &mut ArrayBase<S2, D2>,
        ) -> Result<(), PyErr> {
            #[expect(clippy::unit_arg)]
            if src.shape() == dst.shape() {
                Ok(dst.assign(&src.try_readonly()?.as_array()))
            } else {
                Err(PyValueError::new_err(format!(
                    "mismatching shape {:?} of array-like, expected {:?}",
                    src.shape(),
                    dst.shape(),
                )))
            }
        }

        let ndarray = numpy_asarray(py, array_like)?;

        if let Ok(d) = ndarray.downcast::<PyArrayDyn<u8>>() {
            if let AnyArrayBase::U8(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<u16>>() {
            if let AnyArrayBase::U16(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<u32>>() {
            if let AnyArrayBase::U32(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<u64>>() {
            if let AnyArrayBase::U64(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i8>>() {
            if let AnyArrayBase::I8(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i16>>() {
            if let AnyArrayBase::I16(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i32>>() {
            if let AnyArrayBase::I32(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<i64>>() {
            if let AnyArrayBase::I64(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<f32>>() {
            if let AnyArrayBase::F32(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else if let Ok(d) = ndarray.downcast::<PyArrayDyn<f64>>() {
            if let AnyArrayBase::F64(ref mut view_mut) = view_mut {
                return shape_checked_assign(d, view_mut);
            }
        } else {
            return Err(PyTypeError::new_err(format!(
                "unsupported dtype {} of array-like",
                ndarray.dtype()
            )));
        }

        Err(PyTypeError::new_err(format!(
            "mismatching dtype {} of array-like, expected {}",
            ndarray.dtype(),
            view_mut.dtype(),
        )))
    }
}

impl Clone for PyCodecAdapter {
    fn clone(&self) -> Self {
        #[expect(clippy::expect_used)] // clone is *not* fallible
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
            codec_config_schema: self.codec_config_schema.clone(),
        })
    }

    fn get_config<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Python::with_gil(|py| {
            let config = self
                .codec
                .bind(py)
                .get_config()
                .map_err(serde::ser::Error::custom)?;

            transcode(&mut Depythonizer::from_object(config.as_any()), serializer)
        })
    }
}

/// Wrapper around [`PyCodecClass`]es to use the [`DynCodecType`] API.
pub struct PyCodecClassAdapter {
    class: Py<PyCodecClass>,
    codec_id: Arc<String>,
    codec_config_schema: Arc<Schema>,
}

impl PyCodecClassAdapter {
    /// Wraps a [`PyCodecClass`] to use the [`DynCodecType`] API.
    ///
    /// # Errors
    ///
    /// Errors if the codec `class` does not provide an identifier.
    pub fn from_codec_class(class: Bound<PyCodecClass>) -> Result<Self, PyErr> {
        let codec_id = class.codec_id()?;

        let codec_config_schema = schema_from_codec_class(class.py(), &class).map_err(|err| {
            PyTypeError::new_err(format!(
                "failed to extract the {codec_id} codec config schema: {err}"
            ))
        })?;

        Ok(Self {
            class: class.unbind(),
            codec_id: Arc::new(codec_id),
            codec_config_schema: Arc::new(codec_config_schema),
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

    /// If `class` represents an exported [`DynCodecType`] `T`, i.e. it was
    /// initially created with [`crate::export_codec_class`], the `with` closure
    /// provides access to the instance of type `T`.
    ///
    /// If `class` is not an instance of `T`, the `with` closure is *not* run
    /// and `None` is returned.
    pub fn with_downcast<T: DynCodecType, O: Ungil>(
        py: Python,
        class: &Bound<PyCodecClass>,
        with: impl Send + Ungil + for<'a> FnOnce(&'a T) -> O,
    ) -> Option<O> {
        let Ok(ty) = class.getattr(intern!(class.py(), RustCodec::TYPE_ATTRIBUTE)) else {
            return None;
        };

        let Ok(ty) = ty.downcast_into_exact::<RustCodecType>() else {
            return None;
        };

        let ty: &T = ty.get().downcast()?;

        // The `with` closure contains arbitrary Rust code and may block,
        // which we cannot allow while holding the GIL
        Some(py.allow_threads(|| with(ty)))
    }
}

impl DynCodecType for PyCodecClassAdapter {
    type Codec = PyCodecAdapter;

    fn codec_id(&self) -> &str {
        &self.codec_id
    }

    fn codec_config_schema(&self) -> Schema {
        (*self.codec_config_schema).clone()
    }

    fn codec_from_config<'de, D: Deserializer<'de>>(
        &self,
        config: D,
    ) -> Result<Self::Codec, D::Error> {
        Python::with_gil(|py| {
            let config =
                transcode(config, Pythonizer::new(py)).map_err(serde::de::Error::custom)?;
            let config: Bound<PyDict> = config.extract().map_err(serde::de::Error::custom)?;

            let codec = self
                .class
                .bind(py)
                .codec_from_config(config.as_borrowed())
                .map_err(serde::de::Error::custom)?;

            Ok(PyCodecAdapter {
                codec: codec.unbind(),
                class: self.class.clone_ref(py),
                codec_id: self.codec_id.clone(),
                codec_config_schema: self.codec_config_schema.clone(),
            })
        })
    }
}
