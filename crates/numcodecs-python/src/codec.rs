use pyo3::{
    ffi::PyTypeObject,
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{DerefToPyAny, IntoPyDict, PyDict, PyType},
    PyTypeInfo,
};

use crate::{sealed::Sealed, CodecClass};

/// Represents a [`numcodecs.abc.Codec`] *instance* object.
///
/// The [`Bound<Codec>`] type implements the [`CodecMethods`] API.
///
/// [`numcodecs.abc.Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#module-numcodecs.abc
#[repr(transparent)]
pub struct Codec {
    _codec: PyAny,
}

/// Methods implemented for [`Codec`]s.
pub trait CodecMethods<'py>: Sealed {
    /// Encodes the data in the buffer `buf` and returns the result.
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

impl<'py> Sealed for Bound<'py, Codec> {}

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
