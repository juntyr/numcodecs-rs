use pyo3::{
    ffi::PyTypeObject,
    intern,
    prelude::*,
    types::{DerefToPyAny, PyDict, PyType},
    PyTypeInfo,
};

use crate::{sealed::Sealed, Codec};

/// Represents a [`numcodecs.abc.Codec`] *class* object.
///
/// The [`Bound<CodecClass>`] type implements the [`CodecClassMethods`] API.
///
/// [`numcodecs.abc.Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#module-numcodecs.abc
#[repr(transparent)]
pub struct CodecClass {
    _class: PyType,
}

/// Methods implemented for [`CodecClass`]es.
pub trait CodecClassMethods<'py>: Sealed {
    /// Gets the codec identifier.
    ///
    /// # Errors
    ///
    /// Errors if the codec does not provide an identifier.
    fn codec_id(&self) -> Result<String, PyErr>;

    /// Instantiate a codec from a configuration dictionary.
    ///
    /// # Errors
    ///
    /// Errors if constructing the codec fails.
    fn codec_from_config(
        &self,
        config: Borrowed<'_, 'py, PyDict>,
    ) -> Result<Bound<'py, Codec>, PyErr>;
}

impl<'py> CodecClassMethods<'py> for Bound<'py, CodecClass> {
    fn codec_id(&self) -> Result<String, PyErr> {
        let py = self.py();

        let codec_id = self.as_any().getattr(intern!(py, "codec_id"))?.extract()?;

        Ok(codec_id)
    }

    fn codec_from_config(
        &self,
        config: Borrowed<'_, 'py, PyDict>,
    ) -> Result<Bound<'py, Codec>, PyErr> {
        let py = self.py();

        self.as_any()
            .call_method1(intern!(py, "from_config"), (config,))?
            .extract()
    }
}

impl<'py> Sealed for Bound<'py, CodecClass> {}

#[doc(hidden)]
impl DerefToPyAny for CodecClass {}

#[doc(hidden)]
#[allow(unsafe_code)]
unsafe impl PyNativeType for CodecClass {
    type AsRefSource = Self;
}

#[doc(hidden)]
#[allow(unsafe_code)]
unsafe impl PyTypeInfo for CodecClass {
    const MODULE: Option<&'static str> = Some("numcodecs.abc");
    const NAME: &'static str = "Codec";

    #[inline]
    fn type_object_raw(py: Python) -> *mut PyTypeObject {
        PyType::type_object_raw(py)
    }

    #[inline]
    fn is_type_of_bound(object: &Bound<'_, PyAny>) -> bool {
        let ty = match object.downcast::<PyType>() {
            Ok(ty) => ty,
            Err(_) => return false,
        };

        ty.is_subclass_of::<Codec>().unwrap_or(false)
    }

    #[inline]
    fn is_exact_type_of_bound(object: &Bound<'_, PyAny>) -> bool {
        object.as_ptr() == Codec::type_object_raw(object.py()).cast()
    }
}
