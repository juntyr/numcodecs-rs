use pyo3::{intern, prelude::*, sync::GILOnceCell, types::PyDict};

use crate::{Codec, CodecClass};

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
