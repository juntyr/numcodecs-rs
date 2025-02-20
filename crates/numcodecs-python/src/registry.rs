use pyo3::{prelude::*, sync::GILOnceCell, types::PyDict};

#[expect(unused_imports)] // FIXME: use expect, only used in docs
use crate::PyCodecClassMethods;
use crate::{PyCodec, PyCodecClass};

/// Dynamic registry of codec classes.
pub struct PyCodecRegistry {
    _private: (),
}

impl PyCodecRegistry {
    /// Instantiate a codec from a configuration dictionary.
    ///
    /// The config *must* include the `id` field with the
    /// [`PyCodecClassMethods::codec_id`].
    ///
    /// # Errors
    ///
    /// Errors if no codec with a matching `id` has been registered, or if
    /// constructing the codec fails.
    pub fn get_codec<'py>(config: Borrowed<'_, 'py, PyDict>) -> Result<Bound<'py, PyCodec>, PyErr> {
        static GET_CODEC: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

        let py = config.py();

        let get_codec = GET_CODEC.import(py, "numcodecs.registry", "get_codec")?;

        get_codec.call1((config,))?.extract()
    }

    /// Register a codec class.
    ///
    /// If the `codec_id` is provided, it is used instead of
    /// [`PyCodecClassMethods::codec_id`].
    ///
    /// This function maintains a mapping from codec identifiers to codec
    /// classes. When a codec class is registered, it will replace any class
    /// previously registered under the same codec identifier, if present.
    ///
    /// # Errors
    ///
    /// Errors if registering the codec class fails.
    pub fn register_codec(
        class: Borrowed<PyCodecClass>,
        codec_id: Option<&str>,
    ) -> Result<(), PyErr> {
        static REGISTER_CODEC: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

        let py = class.py();

        let register_codec = REGISTER_CODEC.import(py, "numcodecs.registry", "register_codec")?;

        register_codec.call1((class, codec_id))?;

        Ok(())
    }
}
