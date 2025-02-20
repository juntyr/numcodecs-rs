use pyo3::{
    ffi::PyTypeObject,
    intern,
    prelude::*,
    types::{DerefToPyAny, PyDict, PyType},
    PyTypeInfo,
};

use crate::{sealed::Sealed, PyCodec};

/// Represents a [`numcodecs.abc.Codec`] *class* object.
///
/// The [`Bound<CodecClass>`] type implements the [`PyCodecClassMethods`] API.
///
/// [`numcodecs.abc.Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#module-numcodecs.abc
#[repr(transparent)]
pub struct PyCodecClass {
    _class: PyType,
}

/// Methods implemented for [`PyCodecClass`]es.
pub trait PyCodecClassMethods<'py>: Sealed {
    /// Gets the codec identifier.
    ///
    /// # Errors
    ///
    /// Errors if the codec does not provide an identifier.
    fn codec_id(&self) -> Result<String, PyErr>;

    /// Instantiate a codec from a configuration dictionary.
    ///
    /// The `config` dict must *not* contain an `id` field.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec fails.
    fn codec_from_config(
        &self,
        config: Borrowed<'_, 'py, PyDict>,
    ) -> Result<Bound<'py, PyCodec>, PyErr>;

    /// Gets the [`PyType`] that this [`PyCodecClass`] represents.
    fn as_type(&self) -> &Bound<'py, PyType>;
}

impl<'py> PyCodecClassMethods<'py> for Bound<'py, PyCodecClass> {
    fn codec_id(&self) -> Result<String, PyErr> {
        let py = self.py();

        let codec_id = self.as_any().getattr(intern!(py, "codec_id"))?.extract()?;

        Ok(codec_id)
    }

    fn codec_from_config(
        &self,
        config: Borrowed<'_, 'py, PyDict>,
    ) -> Result<Bound<'py, PyCodec>, PyErr> {
        let py = self.py();

        self.as_any()
            .call_method1(intern!(py, "from_config"), (config,))?
            .extract()
    }

    fn as_type(&self) -> &Bound<'py, PyType> {
        #[expect(unsafe_code)]
        // Safety: PyCodecClass is a wrapper around PyType
        unsafe {
            self.downcast_unchecked()
        }
    }
}

impl Sealed for Bound<'_, PyCodecClass> {}

#[doc(hidden)]
impl DerefToPyAny for PyCodecClass {}

#[doc(hidden)]
#[expect(unsafe_code)]
unsafe impl PyTypeInfo for PyCodecClass {
    const MODULE: Option<&'static str> = Some("numcodecs.abc");
    const NAME: &'static str = "Codec";

    #[inline]
    fn type_object_raw(py: Python) -> *mut PyTypeObject {
        PyType::type_object_raw(py)
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        let Ok(ty) = object.downcast::<PyType>() else {
            return false;
        };

        ty.is_subclass_of::<PyCodec>().unwrap_or(false)
    }

    #[inline]
    fn is_exact_type_of(object: &Bound<'_, PyAny>) -> bool {
        object.as_ptr() == PyCodec::type_object_raw(object.py()).cast()
    }
}
