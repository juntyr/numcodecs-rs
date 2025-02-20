use pyo3::{
    ffi::PyTypeObject,
    intern,
    prelude::*,
    sync::GILOnceCell,
    types::{DerefToPyAny, IntoPyDict, PyDict, PyType},
    PyTypeInfo,
};

#[expect(unused_imports)] // FIXME: use expect, only used in docs
use crate::PyCodecClassMethods;
use crate::{sealed::Sealed, PyCodecClass};

/// Represents a [`numcodecs.abc.Codec`] *instance* object.
///
/// The [`Bound<Codec>`] type implements the [`PyCodecMethods`] API.
///
/// [`numcodecs.abc.Codec`]: https://numcodecs.readthedocs.io/en/stable/abc.html#module-numcodecs.abc
#[repr(transparent)]
pub struct PyCodec {
    _codec: PyAny,
}

/// Methods implemented for [`PyCodec`]s.
pub trait PyCodecMethods<'py>: Sealed {
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
    /// The dict *must* include an `id` field with the
    /// [`PyCodecClassMethods::codec_id`].
    ///
    /// The dict *must* be compatible with JSON encoding.
    ///
    /// # Errors
    ///
    /// Errors if getting the codec configuration fails.
    fn get_config(&self) -> Result<Bound<'py, PyDict>, PyErr>;

    /// Returns the [`PyCodecClass`] of this codec.
    fn class(&self) -> Bound<'py, PyCodecClass>;
}

impl<'py> PyCodecMethods<'py> for Bound<'py, PyCodec> {
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
            Some(&[(intern!(py, "out"), out)].into_py_dict(py)?),
        )
    }

    fn get_config(&self) -> Result<Bound<'py, PyDict>, PyErr> {
        let py = self.as_any().py();

        self.as_any()
            .call_method0(intern!(py, "get_config"))?
            .extract()
    }

    #[expect(clippy::expect_used)]
    fn class(&self) -> Bound<'py, PyCodecClass> {
        // extracting a codec guarantees that its class is a codec class
        self.as_any()
            .get_type()
            .extract()
            .expect("Codec's class must be a CodecClass")
    }
}

impl Sealed for Bound<'_, PyCodec> {}

#[doc(hidden)]
impl DerefToPyAny for PyCodec {}

#[doc(hidden)]
#[expect(unsafe_code)]
unsafe impl PyTypeInfo for PyCodec {
    const MODULE: Option<&'static str> = Some("numcodecs.abc");
    const NAME: &'static str = "Codec";

    #[inline]
    fn type_object_raw(py: Python) -> *mut PyTypeObject {
        static CODEC_TYPE: GILOnceCell<Py<PyType>> = GILOnceCell::new();

        let ty = CODEC_TYPE.import(py, "numcodecs.abc", "Codec");

        #[expect(clippy::expect_used)]
        let ty = ty.expect("failed to access the `numpy.abc.Codec` type object");

        ty.as_type_ptr()
    }
}
