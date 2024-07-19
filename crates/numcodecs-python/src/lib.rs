//! Rust-bindings for the [`numcodecs`] Python API using [`pyo3`].
//!
//! [`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/
//! [`pyo3`]: https://docs.rs/pyo3/0.21/pyo3/

use pyo3::{
    ffi::PyTypeObject,
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{DerefToPyAny, IntoPyDict, PyDict, PyType},
    PyTypeInfo,
};

/// Dynamic registry of codec classes.
pub struct Registry {
    _private: (),
}

impl Registry {
    /// Instantiate a codec from a configuration dictionary.
    ///
    /// The config must include the `id` field with the
    /// [`CodecClassMethods::codec_id`].
    ///
    /// # Errors
    ///
    /// Errors if no codec with a matching `id` has been registered, or if
    /// constructing the codec fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # use pyo3::{prelude::*, types::PyDict};
    /// # use numcodecs_python::Registry;
    ///
    /// Python::with_gil(|py| {
    ///     let config = PyDict::new_bound(py);
    ///     config.set_item("id", "crc32");
    ///
    ///     let codec = Registry::get_codec(config.as_borrowed()).unwrap();
    /// });
    /// ```
    pub fn get_codec<'py>(config: Borrowed<'_, 'py, PyDict>) -> Result<Bound<'py, Codec>, PyErr> {
        static GET_CODEC: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

        let py = config.py();

        let get_codec = GET_CODEC.get_or_try_init(py, || -> Result<_, PyErr> {
            Ok(py
                .import_bound(intern!(py, "numcodecs.registry"))?
                .getattr(intern!(py, "get_codec"))?
                .unbind())
        })?;

        get_codec.call1(py, (config,))?.extract(py)
    }

    /// Register a codec class.
    ///
    /// If the `codec_id` is provided, it is used insted of
    /// [`CodecClassMethods::codec_id`].
    ///
    /// This function maintains a mapping from codec identifiers to codec
    /// classes. When a codec class is registered, it will replace any class
    /// previously registered under the same codec identifier, if present.
    ///
    /// # Errors
    ///
    /// Errors if registering the codec class fails.
    pub fn register_codec(
        class: Borrowed<CodecClass>,
        codec_id: Option<&str>,
    ) -> Result<(), PyErr> {
        static REGISTER_CODEC: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

        let py = class.py();

        let register_codec = REGISTER_CODEC.get_or_try_init(py, || -> Result<_, PyErr> {
            Ok(py
                .import_bound(intern!(py, "numcodecs.registry"))?
                .getattr(intern!(py, "register_codec"))?
                .unbind())
        })?;

        register_codec.call1(py, (class, codec_id))?;

        Ok(())
    }
}

/// Represents a [`numcodecs.abc.Codec`] *instance* object.
///
/// The [`Bound<Codec>`] type implements the [`CodecMethods`] API.
///
/// [`numcodecs.abc.Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#module-numcodecs.abc
#[repr(transparent)]
pub struct Codec {
    codec: PyAny,
}

/// Methods implemented for [`Codec`]s.
pub trait CodecMethods<'py>: sealed::Sealed {
    /// Encode the data in the buffer `buf` and returns the result.
    ///
    /// The input and output buffers be any objects supporting the
    /// [new-style buffer protocol].
    ///
    /// # Errors
    ///
    /// Errors if encoding the buffer fails.
    ///
    /// [new-style buffer protocol]: https://docs.python.org/3/c-api/buffer.html
    fn encode(&self, buf: Borrowed<'_, 'py, PyAny>) -> Result<Bound<'py, PyAny>, PyErr>;

    /// Decodes the data in the buffer `buf` and returns the result.
    ///
    /// The input and output buffers be any objects supporting the
    /// [new-style buffer protocol].
    ///
    /// If the optional output buffer `out` is provided, the decoded data is
    /// written into `out` and the `out` buffer is returned. Note that this
    /// buffer must be exactly the right size to store the decoded data.
    ///
    /// If the optional output buffer `out` is *not* provided, a new output
    /// buffer is allocated.
    ///
    /// # Errors
    ///
    /// Errors if decoding the buffer fails.
    ///
    /// [new-style buffer protocol]: https://docs.python.org/3/c-api/buffer.html
    fn decode(
        &self,
        buf: Borrowed<'_, 'py, PyAny>,
        out: Option<Borrowed<'_, 'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, PyErr>;

    /// Returns a dictionary holding configuration parameters for this codec.
    ///
    /// The dict must include an `id` field with the
    /// [`CodecClassMethods::codec_id`]. The dict must be compatible with JSON
    /// encoding.
    ///
    /// # Errors
    ///
    /// Errors if getting the codec configuration fails.
    fn get_config(&self) -> Result<Bound<'py, PyDict>, PyErr>;

    /// Returns the [`CodecClass`] of this codec.
    fn class(&self) -> Bound<'py, CodecClass>;
}

impl<'py> CodecMethods<'py> for Bound<'py, Codec> {
    fn encode(&self, buf: Borrowed<'_, 'py, PyAny>) -> Result<Bound<'py, PyAny>, PyErr> {
        let py = self.py();

        self.as_any().call_method1(intern!(py, "encode"), (buf,))
    }

    fn decode(
        &self,
        buf: Borrowed<'_, 'py, PyAny>,
        out: Option<Borrowed<'_, 'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, PyErr> {
        let py = self.as_any().py();

        self.as_any().call_method(
            intern!(py, "decode"),
            (buf,),
            Some(&[(intern!(py, "out"), out)].into_py_dict_bound(py)),
        )
    }

    fn get_config(&self) -> Result<Bound<'py, PyDict>, PyErr> {
        let py = self.as_any().py();

        self.as_any()
            .call_method0(intern!(py, "get_config"))?
            .extract()
    }

    #[allow(clippy::expect_used)]
    fn class(&self) -> Bound<'py, CodecClass> {
        // extracting a codec guarantees that its class is a codec class
        self.as_any()
            .get_type()
            .extract()
            .expect("Codec's class must be a CodecClass")
    }
}

impl<'py> sealed::Sealed for Bound<'py, Codec> {}

#[doc(hidden)]
impl DerefToPyAny for Codec {}

#[doc(hidden)]
#[allow(unsafe_code)]
unsafe impl PyNativeType for Codec {
    type AsRefSource = Self;
}

#[doc(hidden)]
#[allow(unsafe_code)]
unsafe impl PyTypeInfo for Codec {
    const MODULE: Option<&'static str> = Some("numcodecs.abc");
    const NAME: &'static str = "Codec";

    #[inline]
    fn type_object_raw(py: Python) -> *mut PyTypeObject {
        static CODEC_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();

        let ty = CODEC_TYPE.get_or_try_init(py, || {
            py.import_bound(intern!(py, "numcodecs.abc"))?
                .getattr(intern!(py, "Codec"))?
                .extract()
        });
        #[allow(clippy::expect_used)]
        let ty = ty.expect("failed to access the `numpy.abc.Codec` type object");

        ty.bind(py).as_type_ptr()
    }
}

/// Represents a [`numcodecs.abc.Codec`] *class* object.
///
/// The [`Bound<CodecClass>`] type implements the [`CodecClassMethods`] API.
///
/// [`numcodecs.abc.Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#module-numcodecs.abc
#[repr(transparent)]
pub struct CodecClass {
    class: PyType,
}

/// Methods implemented for [`CodecClass`]es.
pub trait CodecClassMethods<'py>: sealed::Sealed {
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

impl<'py> sealed::Sealed for Bound<'py, CodecClass> {}

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
        let Ok(ty) = object.downcast::<PyType>() else {
            return false;
        };

        ty.is_subclass_of::<Codec>().unwrap_or(false)
    }

    #[inline]
    fn is_exact_type_of_bound(object: &Bound<'_, PyAny>) -> bool {
        object.as_ptr() == Codec::type_object_raw(object.py()).cast()
    }
}

mod sealed {
    pub trait Sealed {}
}
